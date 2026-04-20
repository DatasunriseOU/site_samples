---
title: "Fused MoE and DeepEP on NVIDIA: the dispatch layer we ship"
description: "How MegaCpp dispatches MoE tokens on H200 and GB10: DeepEP NVSHMEM all-to-all on NVLink and IB, fused expert GEMM, expert sharding, drop policies, and how the kernel layer interacts with our eight-specialist routing."
date: "2026-04-18"
tags: ["moe", "deep-ep", "nvshmem", "all-to-all", "H200", "nvidia", "fused-moe"]
---

If you ship MoE on NVIDIA at any non-trivial expert count, the dispatch layer is the entire performance story. The router is a few percent of the FLOPs and almost none of the wall clock; the all-to-all and the expert GEMM are everything. This post is the NVIDIA-only counterpart to the [routing decisions writeup](/blog/moe-routing-we-actually-shipped) — the same 64-expert / top-6 / 8-specialist setup, but with the focus on how DeepEP, the Megatron-Core flex dispatcher, and the fused expert path actually move tokens on H200 multi-node systems and GB10. Routing policy is taken as given; we are talking about the wires under the floor.

## Why MegaCpp cares about this

An 8-specialist ensemble runs MoE layers at every E-block in a depth-52 hybrid preset. With 64 routed experts, top-6 routing, and EP=4 on a single 8-GPU H200 node (two MoE EP groups co-located with FSDP), the standard `torch.distributed.all_to_all_single` path was attributing roughly a third of the step time to the dispatch / combine pair, with another double-digit slice burned on `FillFunctor` from per-call buffer allocations and `scatter_add_` from the compact-buffer build. The Megatron-Core team had already published the cure in [`fused_a2a.py` — NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM): NVSHMEM-direct dispatch via DeepEP, with EventOverlap handles for true comm/compute overlap, plus the fused permute kernel from Transformer Engine on the local side. That is the path we adapteded.

The constraint that shaped the design is mundane: we have to fall back. CI runs on macOS, ablation paths run on a single H200 without IB, GB10 has no second host, and the offline harness needs to import the MoE module without NVSHMEM in the process. Every layer of the dispatch stack has a non-DeepEP twin, and the choice between them is made at construction time from environment, not at every forward call.

## What we built in the public MegaCpp MoE dispatch path

The MoE substrate in the codebase is three modules deep, plus a bridge.

the main MoE runtime module is the configuration and the soft-routed reference. `MoEConfig` carries the static knobs — `n_routed_experts`, `n_shared_experts`, `top_k`, `expert_size`, `shared_expert_size`, the FP8 alignment constant, the auxiliary-loss coefficients — and the layer ships a fully-soft variant that computes all experts and weights the outputs. The soft variant is XLA-safe and exists for ablation, never for deployment. The interesting bookkeeping is `get_local_expert_indices(n_experts, ep_size, ep_rank)`, which is the contract every dispatcher reads to know what slice of the expert table this rank owns. The router itself runs in fp32 (`_router_fp32_linear`) because sigmoid logits at bf16 produce just enough scoring noise to flip top-k membership across reruns; we kept fp32 routing as a hard rule across every dispatcher variant.

the MoE dispatch runtime module is the `AlltoAllTokenDispatcher`. It implements two modes: a CUDA variable-split path that packs only active tokens contiguously and exchanges with per-rank split sizes via `dist.all_to_all_single`, and an XLA equal-split path that pads every rank's slice to a fixed capacity so the compile graph stays static. The CUDA path is the one that matters for NVIDIA. It owns the `_MoEBufferCache` — a Megatron-style `GlobalMemoryBuffer` clone that pre-allocates flat storage and returns a fresh tensor view per call — to kill the `cudaMalloc` and `FillFunctor` storm we saw in nsys. The differentiable backward goes through `_AlltoAllFunction` (synchronous) or `_AsyncAlltoAllFunction` (overlap-friendly). On top of that scaffold, three opt-in paths layer:

First, the TE permute path. When `MEGACPP_MOE_TE_PERMUTE=1` is set and TE is importable, the dispatcher swaps the `nonzero + index_select + scatter_add_` compact-buffer construction for `te.moe_permute_with_probs`. The unpermute side still goes through the argsort inverse for the foreseeable horizon because the combine path is harder to validate against the local baseline.

Second, the Megatron MoE utils path. the Megatron MoE integration module is a thin wrapper around `megatron.core.transformer.moe.moe_utils` that exposes `permute`, `unpermute`, and `group_limited_topk` with our index/probability conventions. When the Megatron module is importable and the env flag is on, the dispatcher calls into it instead of the local `scatter_add_`. This is the bridge we use on machines that have Megatron installed but not DeepEP.

Third, the DeepEP path. When `MEGACPP_MOE_DEEPEP=1` is set, the dispatcher delegates to `DeepEPDispatcher` from `deep_ep_bridge.py` for the actual exchange.

The public DeepEP bridge wraps `deep_ep.Buffer`. The interesting code is the buffer cache: `Buffer` instances are sized once per process group via `Buffer.get_dispatch_config(ep_size).get_nvl_buffer_size_hint(hidden_bytes, ep_size)` and the analogous combine config, then memoised by `id(group)`. We never re-create a Buffer unless the group changes or the required size grows. This is the same pattern Megatron uses in its `get_buffer()` helper, and it matters because allocating an NVSHMEM slab is not cheap and doing it on every forward would defeat the point. The dispatcher's `dispatch()` returns the canonical 6-tuple `(recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert, handle, event_overlap)`. The `handle` carries the layout DeepEP needs to reverse the exchange in `combine()`; the `event_overlap` is the synchronisation point we defer until just before the combine fires, so the expert GEMM runs in parallel with the dispatch wait.

the fused MoE kernel module is the device-local expert compute. It picks one of three implementations at runtime: a Triton fused kernel that ties top-k routing, sort-by-expert, jagged grouped GEMM, activation, second GEMM, and weighted scatter into one pipeline; a `torch.grouped_mm` persistent grouped GEMM path on torch ≥ 2.10; and a pure PyTorch reference loop. The Triton path is the one we evaluated against; the `_FusedMoEBufferCache` keeps the per-call permute / combine buffers warm for the same reason the MoE dispatch runtime module does. The cuDNN SwiGLU grouped GEMM path on SM100 is wired through `_moeblaze_load_cudnn_grouped_gemm_dswiglu_sm100`; on SM90 we fall back to the Triton + grouped_mm pair.

the Megatron MoE integration module is the Megatron-Core glue mentioned above. The two helpers worth flagging are `permute` and `group_limited_topk`. Group-limited routing — limit routing to a subset of expert groups per token, take top-k within those groups — composes with our 8-specialist hierarchical routing because each specialist owns its own group. We do not invoke it in the main training loop yet (the specialist boundary is enforced higher up), but it is wired through so we can A/B against the global top-k policy when the spec changes.

Drop policy in the public MegaCpp MoE path is simple: capacity-factor drops are off by construction. The variable-split CUDA path sends every token, and the XLA equal-split path pads to a static capacity that is provably larger than the worst-case load (`cap_per_rank = BT * max_active_slots_per_token`). When the auxiliary load-balance loss does its job, the padded headroom is small. When it does not, we surface the imbalance as a metric rather than silently dropping; that decision predates the DeepEP lift and we kept it. DeepEP itself supports a drop mode; we do not enable it.

## How it lands in MegaCpp

The deployment substrate is Megatron-LM with the flex MoE dispatcher.

The Megatron argument bridge builds the argument fragment that turns this on. The relevant chunk: `--moe-token-dispatcher-type flex`, `--moe-router-dtype fp32`, `--moe-permute-fusion`, and conditionally `--moe-grouped-gemm`. The flex dispatcher is Megatron's name for the DeepEP path: pre-allocated fixed buffers instead of the alltoall path that creates very large transient tensors at higher EP, NVLink for intranode transfer, and NVSHMEM IBGDA for multi-node. It silently falls back to the `alltoall` dispatcher if `deep_ep` is not importable, which is the behavior we want on the paths where DeepEP is not installed. `--moe-router-dtype fp32` is non-negotiable because DeepEP only consumes fp32 router probabilities; this matches the public MegaCpp hard rule.

The expert sharding is straightforward. With 64 experts and EP=4, each rank owns 16 experts. The TE `GroupedLinear` path described in the [Transformer Engine bridge](/blog/transformer-engine-bridge-on-nvidia) writeup is what executes the expert GEMM per rank; jagged token counts dispatch as a single fused kernel and the FP8 current-scaling recipe carries through.

The hybrid schedule plan is what makes the dispatch overlap actually work in our hybrid layer mix. Megatron's combined-1F1B schedule plan assumes a stack of pure transformer layers; our model interleaves Mamba M-blocks, MoE E-blocks, and DSA attention layers. `_OpaqueScheduleNode` wraps the Mamba and DSA layers so they execute as opaque nodes on the compute stream while the MoE dispatch / combine on adjacent layers progresses on the comm stream. The MoE layers stay decomposed (`moe_dispatch`, `mlp`, `moe_combine` as distinct schedule nodes) so the all-to-all overlaps with the next layer's compute, the expert GEMM overlaps with the previous layer's combine wait, and the schedule plan keeps two streams busy across the hybrid pattern. Without this patch, the comm stream goes idle every time we hit an M or DSA layer, and the DeepEP win evaporates.

The selective FP8 MoE patch enforces the layer-aware FP8 scope: only E-blocks enter the FP8 compute zone. The MoE side of the dispatch (router fp32 → permute fp8 → grouped GEMM fp8 → activation fp8 → grouped GEMM fp8 → unpermute fp8 → combine bf16) is the only place FP8 fires. Attention and Mamba stay bf16. This survives because the loss curve does not drift the way it does when FP8 enters early-layer attention.

The pieces that did not survive the lift: the local Triton fused MoE kernel from the fused MoE kernel module is *not* what runs in deployment. Megatron's TE-backed grouped GEMM is faster on H200 and integrates with the FP8 global state manager; the Triton path stays as the single-GPU offline kernel and the comparison baseline. The reference `AlltoAllTokenDispatcher` is also not on the main critical path — Megatron's flex dispatcher owns dispatch / combine. We kept both around because they are the only thing that runs on paths without Megatron, and because the metric instrumentation in the reference dispatcher is what we use to debug routing imbalance.

The expert-bank pad/strip helpers (`pad_tokens_per_expert`, `strip_padding`) move with the TE GroupedLinear wrapper into the main path. FP8 tensor cores require splits divisible by 16, and that is true on every path.

## Ablations and what we kept

The wins we kept: DeepEP via `--moe-token-dispatcher-type flex` for any run with EP>1 and a working `deep_ep` install, TE `moe_permute_with_probs` via `--moe-permute-fusion` on the local side of the dispatch, fp32 router probabilities everywhere, FP8 grouped GEMM for the expert compute, the `_MoEBufferCache` pattern (lifted into Megatron's GlobalMemoryBuffer where it already exists), and the layer-aware FP8 scope.

The wins we *did not* keep: Expert Choice routing (broken at autoregressive batch size 1, see the routing post), capacity-factor token dropping (we surface imbalance as a metric instead), and the post-hoc TE Linear walk on top of Megatron's TE spec (double-wrap).

The neutral outcomes: blanket `--moe-grouped-gemm` is essentially a no-op on the TE path because TE GroupedLinear is already the GEMM, but the flag stays on for clarity and for the paths that drop back to Megatron's non-TE GroupedMLP. The `EventOverlap` handle from DeepEP composes cleanly with the hybrid schedule plan; the gain over the synchronous variant is in the 8-15% range on the depth-52 preset, which is the difference between "DeepEP is a nice-to-have" and "DeepEP is the entire point".

The boring engineering: every PR runs the dispatcher A/B harness — local `scatter_add_` baseline, TE permute path, Megatron MoE utils path, DeepEP path — on a 2-rank smoke test. If any of the four paths regresses by more than 2% on token-throughput at the same model snapshot, the PR does not land. This is the only reason the fallback paths still work after a year of moving target on the DeepEP and Megatron sides.

## Production checklist

- DeepEP is enabled via `--moe-token-dispatcher-type flex`. The `deep_ep` package is part of the runtime environment; the CI job proves it imports.
- `--moe-router-dtype fp32` is unconditional. Router logits in bf16 produce non-deterministic top-k membership.
- `--moe-permute-fusion` is unconditional on H200 and GB10. The TE permute kernel is the device-local win.
- `Buffer` instances are memoised per `id(group)`. The bridge never re-creates a Buffer on the critical path.
- The expert bank is FP8 with current scaling; per-expert token counts are padded to multiples of 16 before the TE GroupedLinear call.
- Drop policy is "no drop": variable-split on CUDA, padded equal-split on XLA. Imbalance surfaces as a metric, not as silent token loss.
- The hybrid schedule plan patch is applied; opaque nodes wrap Mamba and DSA layers so the comm stream stays busy through the hybrid pattern.
- Layer-aware FP8: E-blocks only. Attention and Mamba stay bf16.
- Smoke harness covers four dispatch paths every PR: baseline `scatter_add_`, TE permute, Megatron MoE utils, DeepEP. A 2% regression on any of them is a hard stop.
- On multi-node runs, NVSHMEM IBGDA is the transport. On single-node runs, NVLink. The `Buffer` sizing hints handle both via `get_nvl_buffer_size_hint` and `get_rdma_buffer_size_hint`.

## DeepEP dispatch surface

| Path | Transport | Where it wins | Where it loses |
|---|---|---|---|
| Intra-node DeepEP | NVSHMEM over NVLink | H200:8 single host, low-jitter | offers nothing on a 1-GPU box |
| Inter-node DeepEP | NVSHMEM over IB | multi-node MoE training | needs IB topology to actually be IB |
| Megatron-Core a2a fallback | NCCL all-to-all | small expert counts, debugging | jitter at high EP, no overlap with GEMM |
| GroupedGEMM expert path | CUTLASS grouped GEMM | dense per-expert tokens | thin experts under-utilize the SMs |

A representative dispatch shape we use on H200:8:

```python
moe_cfg = MoEConfig(
    num_experts=64,
    top_k=2,
    capacity_factor=1.25,
    dispatch=DeepEP(transport="nvshmem", overlap_combine=True),
    expert_gemm="grouped_gemm",
)
```

## References

- Modules: the main MoE runtime module, the fused MoE kernel module, the MoE dispatch runtime module, the public DeepEP bridge, the Megatron MoE integration module, the Megatron argument bridge, the hybrid schedule plan, the selective FP8 MoE patch, the Transformer Engine expert path, and the Transformer Engine permute path.
- [DeepEP — DeepSeek, GitHub](https://github.com/deepseek-ai/DeepEP)
- [Megatron-Core fused_a2a.py — NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [GShard — Lepikhin et al., arXiv:2006.16668]
- [Megablocks — Gale et al., arXiv:2211.15841]
- [Tutel — Hwang et al., arXiv:2206.03382]
- [DeepSeek-V3 — arXiv:2412.19437]
- [vLLM fused_moe — vllm-project, GitHub](https://github.com/vllm-project/vllm)
- [SGLang fused_moe — sgl-project, GitHub](https://github.com/sgl-project/sglang)
