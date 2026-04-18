---
title: "Hybrid FSDP/DDP on NVIDIA: Megatron DDP plus FSDP2 for the Specialist Ensemble"
description: "How MegaCpp combines Megatron-Core DistributedDataParallel with PyTorch FSDP2 across H200 and GB10, the gradient-bucket sizing rules we ship, the freeze plan for the eight specialists, and the failure modes that taught us the contract."
date: "2026-04-18"
tags: ["FSDP2", "Megatron", "Distributed Training", "H200", "GB10"]
---

MegaCpp ships eight specialist networks behind one router. On NVIDIA we do not run them under a single sharding strategy. The base trunk and the dense specialists run under Megatron-Core DistributedDataParallel with the Megatron DistributedOptimizer; the wide MoE specialists and any LoRA-adapted heads run under PyTorch FSDP2 (`fully_shard`). This post is the NVIDIA-specific story: which lane owns which parameters, how we size buckets on H200 and GB10, the freeze plan we ship for the eight specialists, and the failure modes that forced us to stop pretending one wrapper was enough. The cross-lane FSDP2 post covers the abstract memory math and the XLA side; here we stay on CUDA.

## Why MegaCpp cares about this

Two facts dictate the design. First, the production training stack inherits Nemotron Nano 3's parallelism layout almost verbatim, which means Megatron-Core's `DistributedDataParallel` plus `DistributedOptimizer` on top of TP, PP, EP, and CP. That code path is ruthlessly tuned for steady-state throughput on H200 and is the only path Megatron's TensorParallel Muon will accept. Second, several specialists do not match that mold. The wide-MoE specialists need expert sharding strategies that Megatron's expert-parallel buffers do not express well at our sizes; the LoRA-adapted reasoning specialist needs ignored-parameter handling for tiny rank-4 matrices that should never enter an all-gather. FSDP2's `fully_shard` API gives us both, at the cost of leaving the Megatron optimizer behind.

So we run a hybrid. Each specialist is wrapped by exactly one of the two stacks, never both. The trunk uses Megatron DDP. The cross-specialist gradients flow through whatever the specialist's wrapper exposes. Picking the wrapper per specialist instead of per cluster is the single biggest decision in this post; everything else is a consequence.

## What we built in the POC

The two implementation surfaces in the research repo are `megatron_ddp.py` and `fsdp_cuda.py`, with `dtensor_utils.py` providing the DTensor plumbing they share and `fsdp.py` carrying the XLA SPMD analogue we explicitly do not use here.

`megatron_ddp.py` is a thin shim. `wrap_model_with_megatron_ddp()` builds a `TransformerConfig` from our `GPTConfig` (via the Megatron bridge), wraps the model in a `_MegatronConfigInjector` so Megatron's `get_model_config()` discovers a real `TransformerConfig` rather than our native one, and constructs the DDP on a side CUDA stream so it composes with CUDA graphs. Before construction we call `_mark_expert_parallel_params()`, which walks the module tree, finds `FusedExpertBank` / `ExpertMLP` / `FusedMoEExpertBank` modules and any submodule under a `.experts.` path, and stamps `param.allreduce = False` on their parameters. Megatron DDP reads that attribute to split params into dense buffers (allreduced over the data-parallel group) and expert-parallel buffers (reduce-scattered over the expert-data-parallel group). Forget this and the expert grads silently land in the wrong process group.

`build_ddp_config()` carries the Nemotron Nano 3 defaults: `overlap_grad_reduce=True`, `overlap_param_gather=True`, `use_distributed_optimizer=True`, `check_for_nan_in_grad=True`, `average_in_collective=True`, and crucially `grad_reduce_in_fp32=False`. Promoting gradient reduction to fp32 costs roughly 9.5 GB of HBM at our shape and never moved a loss curve we could measure. The optimizer side is `build_megatron_distributed_optimizer()`, which goes through `get_megatron_optimizer` so the TensorParallel Muon path lands automatically for 2D weight matrices and an AdamW handles scalars and embeddings. The training adapter (`MegatronDDPTrainingAdapter`) bridges Megatron's `step()` / `set_lr()` / `no_sync()` to our existing loop.

`fsdp_cuda.py` is the harder file. `apply_cuda_fsdp()` wraps bottom-up: each transformer block first, then embeddings and `lm_head`, then any root-level auxiliary modules (MTP, scalar gates, residual lambdas), then the root last so that optimizer step sees gathered params. Each wrap uses `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`. LoRA awareness is explicit: `_collect_lora_params()` finds every `lora_A` / `lora_B` tensor and passes them as `ignored_params` to `fully_shard`, so rank-4 matrices stay replicated instead of triggering pointless all-gathers; if LoRA is injected after FSDP wrap, `register_lora_grad_hooks()` installs the manual all-reduce hooks since FSDP only manages params that existed at wrap time.

Two knobs deserve calling out. `_block_reshard_after_forward_value()` resolves the per-block `reshard_after_forward` from environment: default `True` (re-all-gather in backward), but settable to `False` (keep unsharded params in HBM after forward, skipping one collective per block per step at the cost of peak memory) or to an integer N for intra-node hybrid sharding to a smaller mesh. `_block_prefetch_limit()` controls how many block all-gathers we let FSDP issue ahead from CPU; values above 1 are aggressive overlap with more reserved memory.

The pile of `_patch_*` helpers above the public API exists because torch nightlies between 2.11 and 2.12 broke FSDP2 on H200 in three independent ways: a private-attribute regression on `FSDPParam.to_accumulated_grad_if_needed`, a redundant device-mesh concatenation in `_init_sharded_param`, and an overlap-mesh concat that left the root wrapper in a torn state when TP and FSDP shared a mesh. `_apply_fsdp2_nightly_workarounds()` is the single boot-time patch hook; the per-call code logs which patches are active so receipts are not ambiguous.

`dtensor_utils.py` carries the safe `to_local_if_dtensor` and `_describe_dtensor_debug` helpers that both lanes use to avoid materializing global tensors when only the local shard is needed. The existence of this module is the clearest signal that Megatron and FSDP had to learn to share one DTensor representation in our code; otherwise their DTensor handling diverges and gradient hooks see different objects.

## How it lands in MegaCpp

In production we consolidate. `megatron_ddp.py` lifts almost as-is into the MegaCpp Megatron integration, since it is already a thin wrapper over upstream. The DDP config builder becomes the single source of truth for our DDP defaults, and `_mark_expert_parallel_params()` becomes a hard precondition checked at boot rather than a best-effort walk; an unmarked expert parameter aborts the run instead of silently joining the dense bucket.

`fsdp_cuda.py` is the file that gets the most aggressive simplification. The nightly workarounds compile out behind a `torch>=2.12.1` floor in MegaCpp. The `_PatchableCallable` and Protocol shims survive only inside a `TYPE_CHECKING` block so we keep static typing without the runtime overhead. The bottom-up wrap walk and the LoRA ignored-param logic stay; they are the two pieces every downstream user has to get right.

The bucket-sizing math from `megatron_optimizer.py:get_default_bucket_size_mb` graduates into the Megatron config: we keep Megatron's own rule of `max(40M, 1M * dp_world_size)` parameters per bucket, but pin the gradient dtype at bf16 so the byte budget is around 80 MB for `dp <= 40` and around 128 MB at `dp = 64`. The April 2026 nsys profile on the depth-52 hybrid preset at EP=2 showed NCCL all-gather as a meaningful slice of step time; bigger buckets amortize launch overhead enough to claw it back without changing the math. We expose `--megatron_bucket_size_mb` as a feature flag for profiling but do not encourage tuning it.

`_block_reshard_after_forward_value` becomes a config field, not an env knob. Same for `_block_prefetch_limit`. The hybrid mesh sharding (the "integer N" path) is the only way we have made FSDP2 cheaper at large DP counts without giving up backward all-gather, and we ship it as a first-class option.

What we drop: the standalone XLA `apply_fsdp_sharding` from `fsdp.py` does not enter MegaCpp. The XLA path is owned by the JAX/Pallas branch and is documented separately. Inside the NVIDIA package, FSDP2 is the only sharding-by-replication wrapper.

What moves to a kernel path: nothing in the wrappers themselves, but the gradient bucket reduce-scatter compiles out to NCCL primitives and we pin the NCCL algorithm per SKU. That is the only piece of NCCL tuning that survived contact with both H200 and GB10.

## Ablations and what we kept

The freeze plan for the eight specialists is the boring engineering. Each specialist has a list of modules that are frozen during its specialist-phase training: the embedding, the trunk transformer blocks below the specialization point, the routing head when not under training, and the LoRA-adapter base weights. FSDP2 handles frozen params correctly out of the box (the changelog has entries from earlier FSDP1 days where it did not). Megatron DDP handles frozen params via the `allreduce` attribute as well; what we had to fix is that frozen params still need to be visible to the optimizer's parameter scan, otherwise late-added groups (the NCP, TOP, and GateSkip heads) miss their LR schedule. The fix is one rule: freeze in-place, do not `del` the parameter from the module.

Bucket sizing on H200 is dominated by NVLink contention rather than HBM. We tried 25 MB, 80 MB, 128 MB, and 256 MB on the depth-52 preset. 80 MB and 128 MB are within noise; 25 MB loses a few percent of step time to NCCL overhead; 256 MB starts to hit param-gather stalls when EP=2 narrows the DP group. The Megatron default plus our bf16 grad rule lands inside the safe band by construction.

On GB10 the picture flips. The unified-memory architecture punishes large pinned bucket buffers because they steal CPU bandwidth from the surrounding training. We cap bucket size at 64 MB on GB10 regardless of what the formula returns, and we disable `pad_buckets_for_high_nccl_busbw` which is meaningless without high-bus-bandwidth NVLink between Spark units. These two GB10 carve-outs are the only places where the NVIDIA lane forks by SKU.

What we tried and dropped:

- A single-wrapper world where everything ran under FSDP2. It worked on small models but the TensorParallel Muon path inside Megatron is too valuable to give up; we lost more in optimizer quality than we gained in wrapper simplicity.
- Disabling `overlap_param_gather` on the theory that param-gather is the loud step. It was loud, and disabling overlap made it louder; throughput regressed cleanly.
- `grad_reduce_in_fp32=True` for "safety". It costs HBM, costs bandwidth, and never moved a loss curve we could measure.
- A custom prefetch scheduler in front of FSDP2. The PyTorch defaults plus our `prefetch_limit` knob were strictly better than what we hand-rolled; the only place hand-tuning still wins is dual-pipe stages, where `apply_cuda_fsdp_to_dualpipe_stages` keeps a separate scheduler for stage boundaries.

The failure modes that taught us the rules:

- An H200 first-forward hang inside block 0 when whole-block FSDP eagerly grouped TP-sharded attention/MLP weights into one unshard. Fix: keep TP block params out of FSDP for the validated TP lanes (`tp_boundary_modules_as_root_only=True`).
- A nested-root wrap deadlock under torch 2.12 nightlies after the overlap-mesh workaround was active. Fix: install a no-param root state so FSDP lazy-init still tracks the root, while real params live on child wrappers and the remaining tiny unmanaged params get manual grad sync via `_collect_unmanaged_param_names()`.
- Silent expert-grad routing to the dense process group when the MoE module class was renamed during a refactor and the marker walk missed it. Fix: the marker is now a hard assertion in production.

## Production checklist

- One wrapper per specialist; never wrap the same module under both Megatron DDP and FSDP2.
- Mark expert parameters with `allreduce=False` before constructing Megatron DDP; assert the count is non-zero on any MoE specialist.
- Use bf16 gradient reduction; do not enable `grad_reduce_in_fp32` without a measured loss-quality reason.
- Set bucket size from the Megatron formula on H200; cap at 64 MB on GB10.
- Pass LoRA params as `ignored_params` to `fully_shard` if injected pre-FSDP; otherwise install `register_lora_grad_hooks` after injection.
- Construct Megatron DDP on a side CUDA stream so it composes with CUDA graphs.
- Wrap FSDP2 bottom-up: blocks, then embeddings/lm_head, then auxiliaries, then root last.
- Prefer `reshard_after_forward=True` by default; flip to integer N for intra-node hybrid sharding only when DP is large.
- Pin NCCL algorithm per SKU rather than relying on autotune.
- Freeze in place; never delete parameters that the optimizer scan must see.

## Specialist freeze and wrapper map

| Specialist | Wrapper | Frozen at specialist phase | Notes |
|---|---|---|---|
| Trunk + dense reasoning | Megatron DDP + DistributedOptimizer | embedding, lower trunk | `allreduce=True`, bf16 grad reduce |
| Wide MoE | FSDP2 + AdamW | embedding, routing head when off | `allreduce=False` on every expert param |
| LoRA-adapted reasoning | FSDP2 + ignored_params | base linear, embedding | rank-4 adapters never enter all-gather |
| MTP / scalar heads | FSDP2 root-level | none | wrapped after blocks, before root |

The contract that prevents the silent expert-grad bug:

```python
def _mark_expert_parallel_params(model):
    for name, mod in model.named_modules():
        if mod.__class__.__name__ in {"FusedExpertBank", "ExpertMLP",
                                      "FusedMoEExpertBank"} or ".experts." in name:
            for p in mod.parameters(recurse=False):
                p.allreduce = False  # routes to expert-data-parallel group
```

## References

- `megatron_ddp.py`, `fsdp_cuda.py`, `dtensor_utils.py`, `fsdp.py`, `megatron_optimizer.py`, `megatron_bridge.py`
- [PyTorch FSDP2 (`fully_shard`) documentation — pytorch.org]
- [Megatron-Core DistributedDataParallel and DistributedOptimizer — NVIDIA Megatron-LM]
- [Nemotron Nano 3 30B-A3B training recipe — NVIDIA technical report]
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — Rajbhandari et al., SC20]
