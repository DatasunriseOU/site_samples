---
title: "Communication cost and overlap: NCCL on H200, XLA collectives on TPU v6e"
description: "How MegaCPP budgets all-reduce, reduce-scatter, and all-gather against compute on the hybrid stack, including bucket sizing, launch coalescing, alignment, and the overlap windows that actually matter."
date: "2026-04-18"
tags: ["nccl", "xla", "fsdp", "performance", "h200", "tpu-v6e"]
---

Distributed training on a hybrid CUDA and TPU stack is a communication problem first and a compute problem second. Dense and MoE configurations live well inside the regime where a careless bucket boundary can erase a large share of step throughput, and where one mis-tuned communication setting on a multi-host H200 lane decides whether a reduce-scatter overlaps the backward pass or stalls it. This post focuses on what the communication cost looks like on the two backends, how bucket sizing and launch coalescing affect overlap, what alignment buys in practice, and which ideas survive the move from prototype to production.

## Why MegaCPP cares about this

Our hybrid stack runs on two very different fabrics. On NVIDIA we target H200:8 single-host with NVLink/NVSwitch as the bread-and-butter rung, and 4xH200:8 multi-host as the upper bound for any preset that does not already burn its budget on memory. On Google we target TPU v6e-8 and v6e-32, where collectives are issued by the XLA SPMD compiler and the unit of optimization is graph shape rather than NCCL call ordering. The dense path is comms-bound past dp=8 with bf16 grads. The MoE path is comms-bound from the very first step because expert parallelism adds an AlltoAll on top of the AllReduce/ReduceScatter we already pay for FSDP-style sharding.

The brief is simple: every microsecond of compute that is not also a microsecond of communication is wasted, and every collective that we cannot hide behind compute is a tax on every step for the rest of the run. The implementation is less simple, because it splits across PyTorch's NCCL bindings, the FSDP2 wrapping logic, our Megatron-style overlapped reducer, and the XLA SPMD lowering on the TPU side.

## What the prototype taught us

On the CUDA path, the central idea is a Megatron-style distributed optimizer with explicit gradient buckets, an overlapped reducer, and a per-bucket all-gather ladder that keeps the next parameter shard in flight by the time the previous wait returns. The chain is built once at initialization and then walked from model hooks so communication is launched from one place instead of racing between multiple call sites.

Bucket size is the first knob. The public default bucket-size helper mirrors Megatron's `max(40_000_000, 1_000_000 * dp_world_size)` parameters formula, multiplied by dtype bytes. At dp=8 that lands around 76 MB; at dp=64 around 122 MB. The old static 40 MB ceiling we used to ship was leaving busbw on the table at high dp because the ring reduce-scatter never reached its asymptote before the next bucket fired.

The second knob is alignment. Padding the flat gradient buffer to a multiple of `world_size * 65536` makes each per-rank shard end on a 64K-element boundary. That mirrors a well-known Megatron-style hint: NCCL ring reduce-scatter on NVLink and NVSwitch tends to hit peak bandwidth on shards that are clean multiples of 64K elements, while leftover tails slow the last chunk. In practice, the lesson is simple: shard alignment is not cosmetic. It changes achieved bandwidth.

The third knob is launch coalescing. Wrapping per-bucket reduce-scatter calls in a coalescing layer lets NCCL fuse multiple kernel dispatches into one launch. That usually helps, but it is still worth keeping a rollback path because communication regressions in specific framework builds do happen.

The fourth knob is grad dtype. `--grad_reduce_in_fp32` allocates `_flat_grad`/`_flat_grad_shard` in fp32, up-casts bf16 grads on copy, runs the reduce-scatter as fp32 AVG, then casts back to `param.dtype` on `write_reduced_grads_to_params`. For our fp32 master-grad optimizer paths (AdamW, Muon) the down-cast is a no-op. The trade is bandwidth: fp32 reduce-scatter doubles the wire bytes versus bf16, so we only flip it on for the dense lane where reductions in low precision were measurably perturbing loss curves on long runs.

On the FSDP2 lane, the same overlap intent shows up through prefetching. Disabling reshard-after-forward and allowing a small prefetch window issues more all-gathers early, at the cost of memory. On H200, a prefetch limit of two was a practical ceiling for the shapes discussed here.

On the TPU path the picture is different because we do not call collectives by hand. The public TPU FSDP notes document the contract: SPMD ZeRO-3 via XLA, where the compiler inserts the all-gather of sharded parameters before forward, the reduce-scatter of gradients after backward, and the optimizer state stays sharded across the data axis. We do not get to hand-tune launch ordering; we get to shape the graph so XLA can. That means avoiding host-device syncs such as `.item()` and `.nonzero()` in hot paths, avoiding data-dependent shapes that would break SPMD propagation, and explicitly replicating tensors that would otherwise pick up the wrong sharding. The MoE path on TPU adds AlltoAll on the expert axis, and the planner has to respect user-pinned tensor and expert parallel choices when the chosen mesh fits.

## How it lands in production

The lift into production is a careful subset. The Megatron-style overlap discipline carries over directly: the production trainer drives a distributed optimizer that already knows the bucket-chain semantics, the reset points, and the alignment rules. What remains at the recipe level is discipline: pin the overlap flags, keep the normalization runtime settings compatible with collective scheduling, and export the small set of NCCL defaults that consistently help on H200.

Three things are being rewritten on the way in. First, a hand-rolled coalescing wrapper retires in favor of an upstream coalesced reduce-scatter path. Second, the bucket-alignment flag becomes a recipe default rather than a per-run knob, because we have not found a workload that benefits from turning it off. Third, the FP32 gradient-reduction choice becomes part of the precision recipe rather than a standalone performance knob, so it composes cleanly with mixed-precision settings.

The two parallelism modes in the production recipe are explicit about the trade. The `nemo_native` mode (TP=2, SP=True) uses Megatron's built-in Mamba mixer and gets full TP-comm overlap. The `author_dp` mode (TP=1, PP=1, DP=8) uses our selective mixer with the Mamba3/M2RNN extensions but accepts a slightly lower comm-overlap potential because there is no TP collective to overlap with the next layer's compute. The recipe documents this trade in plain English; we do not pretend it does not exist.

On TPU, the production lift is mostly graph hygiene. The XLA SPMD path does not change shape; what changes is that we treat `.item()`, `.nonzero()`, and any data-dependent control flow as a hard CI failure on the TPU lane. A safe fallback helper pattern is the template: on XLA the fallback branch is taken, no host-device sync is forced, and no graph break is induced. The TPU comm cost story is then determined entirely by the SPMD-inserted collectives, and our job is to make sure the graph the compiler sees is the graph we intended.

## Ablations and what we kept

The rough expectation on H200 was that aggressive FSDP2 overlap plus launch coalescing could buy a noticeable step-time improvement, while a smaller set of communication-friendly runtime settings added a few more percent on top. It is better to keep that claim qualitative unless you have side-by-side traces for attribution.

What survived contact with real hardware:

- Bucket size formula `max(40M, 1M * dp_size)` parameters times dtype bytes — kept on every CUDA preset.
- 64K-element shard alignment — kept by default; the startup log should confirm it.
- `NCCL_P2P_NET_CHUNKSIZE=524288` and `TORCH_NCCL_HIGH_PRIORITY=1` — kept; both were ported from Megatron-Bridge's `PERF_ENV_VARS`.
- Launch coalescing around per-bucket reduce-scatter launches — kept in the prototype, later retired in later in favor of the upstream path.
- `reshard_after_forward=False` + prefetch limit 2 on FSDP2 — kept on H200:8 with these preset shapes; we did not push the limit higher because memory headroom for the dense+MoE config is tight.

What did not survive:

- The static 40 MB bucket ceiling (replaced by the dp-scaled formula).
- Hand-rolled allreduce calls on the XLA path (`xla_all_reduce_gradients` is gone; SPMD owns it).
- Trying to `bool(torch.any(...).item())` inside Mamba's forward dispatch on XLA (every such call serialized a step into a host sync).
- The older MoE permute path that synchronized token-count metadata to the host — replaced with a cleaner device-side path.

The multi-host story is the most painful one to characterize cleanly. Allreduce stragglers — one rank 100-300 ms slower than the others, often correlating with a specific PCIe slot — show up as the median step time creeping up while the fast ranks idle on the collective. We do not have a magic fix; the playbook is the boring one: pin the slow rank, swap the bench, and re-run. The collective hangs and watchdog tuning are covered separately in the NCCL hangs post; what matters here is that comm cost on multi-host is a distribution, not a number, and the tail dominates.

## Public checklist

- Pin `pad_buckets_for_high_nccl_busbw=True` and confirm the rank-0 announce fires on every CUDA run.
- Use the dp-scaled bucket size formula from the public distributed-optimizer helper; do not hardcode bucket size unless profiling a specific config.
- Set `NCCL_P2P_NET_CHUNKSIZE=524288`, `TORCH_NCCL_HIGH_PRIORITY=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `TORCH_NCCL_AVOID_RECORD_STREAMS=1` on every CUDA rank by default.
- On FSDP2, default to `reshard_after_forward=False` and `prefetch_limit=2`; raise only if memory headroom permits.
- On TPU v6e, treat any new `.item()`, `.nonzero()`, or data-dependent shape in a hot path as a CI failure on the XLA lane.
- Keep `--grad_reduce_in_fp32` opt-in per precision recipe; do not flip it globally.
- Ship the `_coalescing_manager` opt-out env var until the upstream coalesced-collective path is clean on every wheel we support.
- Record the comm-overlap flag set in the run summary so historical comparisons stay honest.

## References

- Public distributed-optimizer, FSDP, tensor-parallel, MoE dispatch, precision-bridge, recurrent-block, and model-wiring documentation in the MegaCPP codebase
- public recipe notes for the public training configuration
- public change notes covering bucket alignment, bucket-size formulas, FSDP2 prefetch wiring, and NCCL defaults
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism — Shoeybi et al., arXiv:1909.08053]
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel — Zhao et al., VLDB 2023]
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs — Xu et al., arXiv:2105.04663]
