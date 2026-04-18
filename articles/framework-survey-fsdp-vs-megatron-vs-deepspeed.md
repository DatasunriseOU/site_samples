---
title: "Framework Survey: FSDP2 vs Megatron-Core vs DeepSpeed vs Torchtitan vs Nanotron vs MaxText"
description: "Honest comparison of the large-scale training frameworks we touched, what each is good at in 2026, and which we use for which lane: NVIDIA training, TPU training, and MegaCpp production."
date: "2026-04-18"
tags: ["fsdp2", "megatron-core", "deepspeed", "torchtitan", "nanotron", "maxtext", "frameworks"]
---

We have shipped training code on PyTorch FSDP2, Megatron-Core, torchtitan-shaped patterns, DeepSpeed (historically), and MaxText-derived flags on TPU. After two years of this, we are qualified to have an opinion. This post is the honest map of which framework is good at what in 2026, what the gaps are, and why our three lanes (NVIDIA pre-training, TPU pre-training, MegaCpp production) use different stacks on purpose.

## Why MegaCpp cares about this

Framework choice is a tax on everything that comes after. The wrong choice means you spend months reimplementing sequence-parallel norm grad sync, or diagnosing silent FP8 scope mismatches, or rewriting your data loader when you move from a research mesh to a production mesh. The right choice means the boring distributed-systems machinery is already solved and you can spend your time on architecture. We have bled enough on each of these frameworks to pick deliberately per lane rather than picking one and hoping.

## What we built in the POC

The research repo supports three distributed paths that live side by side:

Our TPU sharding runtime is XLA SPMD ZeRO-3 for TPUs. It walks the model, builds a per-parameter sharding spec over a `("data", "model", "expert")` mesh, and hands those annotations to `xs.mark_sharding`. The XLA compiler inserts the required all-gather, reduce-scatter, and all-to-all collectives during compilation. There is no step-time comm code - the collectives are HLO ops. This is why `xla_all_reduce_gradients()` is explicitly called out as legacy in our TPU runtime contract: SPMD already handles it.

Our CUDA FSDP2 wrapper uses `torch.distributed.fsdp.fully_shard`. It runs after TP wrapping, before compile, and before the optimizer. It has the usual FSDP2 knobs - `reshard_after_forward`, per-block prefetch limit - and an extra code path for LoRA: if LoRA injection happens after `fully_shard`, the fresh LoRA params are not FSDP-managed and need explicit grad all-reduce hooks, which `register_lora_grad_hooks()` installs.

the distributed parallelism module composes PP + FSDP2 + EP (optionally with TP as a fourth axis) for CUDA runs that need 3D parallelism. It orders the wrapping `TP -> PP -> FSDP2 -> EP` explicitly, which is the torchtitan convention. This is the heaviest-weight path and we keep it because PP is the only real answer for models that do not fit in a single DP stage on H200/GB200.

The Megatron bridge, block wrapper, M2RNN wrapper, MoE integration layer, TP adapter, DDP wrapper, and optimizer wrapper are the Megatron-Core integration. See our separate post on that integration for details; for this survey it is enough to say we use Megatron-Core when we want Transformer Engine kernels, TP/EP communication overlap, and distributed optimizer with bucket coalescing, and we adapt rather than replace.

On the TPU side, our XLA flag bundle is directly descended from MaxText's public flag library, adapted for `torch_xla`. The VMEM limit values, Continuation Fusion settings, and SparseCore offload flags are all empirically tuned in a way only Google and a handful of serious users know. We inherited the MaxText flag taxonomy because rediscovering it from libtpu source code would have cost us months.

We do not currently ship any DeepSpeed code path in the repo. We evaluated it early, kept a few patterns (bucketed gradient reduction, the concept of a distributed optimizer), and moved the implementation onto Megatron-Core's shape instead.

## How it lands in MegaCpp

The production stack is narrower. MegaCpp is a Megatron-Core-native codebase that lives inside the NVIDIA pipeline: the family-aware Megatron entrypoint, the public authored Mamba spec sample, the public recurrent spec sample, the public tensor-parallel Mamba mixer sample, and the `ModuleSpec` family are all written against Megatron's abstractions directly. There is no FSDP2 path in the production NVIDIA lane; Megatron DDP with distributed optimizer is the production choice there.

The TPU lane in production inherits the POC's XLA SPMD path essentially unchanged. We pinned `torch 2.9.0a0+git21fec65`, `torch_xla 2.9.0+gitc04e61c`, `libtpu 0.0.36`, and `jax 0.9.0` per the TPU setup contract, and the mesh shapes are `("data",)`, `("data", "model")`, `("data", "expert")`, or `("data", "expert", "model")` depending on TP and EP sizes. `torch.compile` is explicitly disabled on XLA; the TPU compile contract is per-micro-step `torch_xla.compile()` around fwd+bwd.

## Ablations and what we kept

**PyTorch FSDP / FSDP2.** FSDP2 is the upstream-blessed path now. Meta's PyTorch team documents it as the recommended direction for new training code, with per-parameter DTensor sharding and a cleaner migration story than FSDP1. For our CUDA lane it is the baseline when we do not need TP or PP: a single-node H200 dense run, or a small MoE where EP=1. The good: it composes with `torch.compile` by default in 2.4+, the mp_policy story (param/reduce/buffer dtypes separately controlled) works, and FSDP2 + FP8 via `torchao.Float8Linear` is ergonomically clean if you wrap in the right order. The bad: it does not give you TP or PP; you compose those yourself. For anything past ~1-node dense we end up on Megatron-Core.

**Megatron-Core.** The ceiling for NVIDIA training. Transformer Engine integration is where MFU comes from on Hopper and Blackwell: `TELayerNormColumnParallelLinear` + `TERowParallelLinear` + `TEDotProductAttention` with userbuffer comm-overlap, fused RoPE, fused masked softmax, and a working `fp8_autocast` recipe. TP and EP communication overlap is not theoretical. The distributed optimizer buckets gradients by parameter count with NVLink/NVSwitch-aware padding, coalesces reduce-scatters through `_coalescing_manager`, and overlaps param gather with compute. The cost: it is NVIDIA-shaped. `TransformerConfig` is a Procrustean bed for hybrid architectures (see our Megatron porting post), and features without analogues - expert_choice, null_rho routing, attn_softcap, mHC residual streams - either need adapter work or do not port.

**DeepSpeed.** ZeRO-1/2/3, 3D parallelism, ZeRO-Infinity, and MoE primitives in one package. It is genuinely productive for teams that inherited it early, and the ZeRO offload paths are still competitive for memory-bound experiments. Where it is not our first pick in 2026: the moment you want TE fused kernels and FP8, you are gravitating back to Megatron-Core; the moment you want the cleanest upstream-PyTorch integration, you are on FSDP2. DeepSpeed sits in between and does not decisively win either axis for our workloads. We respect it, we are not running it.

**torchtitan.** Not a framework so much as a reference implementation of how to do FSDP2 + TP + PP + Float8 + compile in native PyTorch, without a framework wrapper. Our the distributed parallelism module follows its wrapping order (TP -> PP -> FSDP2 -> EP). The an FP8 torchao comparison note entry in the research repo explicitly cites torchtitan's FSDP+compile+Float8 = 48% speedup pattern as the template we match. Use torchtitan as a pattern source, not as a dependency.

**nanotron.** Minimalist 3D-parallel PyTorch library from Hugging Face. We looked at it seriously for the small-scale research loop; it is readable and teaches the abstractions well. It is not where production throughput on H200 clusters lives today, and it has no FP8 or TE story to speak of. Good teaching material, not our production lane.

**MaxText.** The JAX/TPU flagship. Production-grade on TPU v5e/v5p/v6e/v7, written against JAX and XLA from the start, and the source of most of the real-world libtpu tuning knowledge in the open-source world. Our TPU flag bundle is a descendant, which is the right level of dependency: we inherit the flag taxonomy and the value ranges, we do not take on JAX. Trying to run MaxText directly would have meant rewriting our model in JAX, which is a non-starter for a codebase that also runs on CUDA.

**T5X.** Mature JAX research framework, mostly seq2seq and Google-research-shaped workflows. It is not where frontier decoder-only LLMs train in 2026 - MaxText has absorbed that role on the JAX side. We have not used it and do not plan to.

The ablations that stuck in our CHANGELOG are mostly cross-framework pattern transfer: bucket-size defaults mirroring Megatron's `max(40M, 1M * dp_world_size)` rule, `grad_reduce_in_fp32` from Megatron's `DDPConfig`, `pad_buckets_for_high_nccl_busbw` for NVLink alignment, `MEGACPP_FSDP2_NO_RESHARD_AFTER_FWD=1` and `MEGACPP_FSDP2_PREFETCH_LIMIT=N` for torchtitan-style aggressive overlap, and MaxText's VMEM limits and continuation-fusion flags for TPU. The bigger lesson: no single framework is complete for a hybrid architecture. We steal one good idea per framework.

## Lane assignments

Here is how we actually split the work in 2026:

NVIDIA pre-training lane: Megatron-Core is the host, TE kernels do the compute, Megatron DDP with the distributed optimizer does the comm. the distributed parallelism module is the fallback for runs where we deliberately want to stay native-PyTorch (FSDP2 + PP + EP) - usually for architecture-debugging where we need finer-grained control than Megatron gives.

TPU pre-training lane: `torch_xla` 2.9 with SPMD ZeRO-3 over a mesh that scales to `("data", "expert", "model")`. Our MaxText-descended flag bundle is applied before `torch_xla` import. `XLAAdamW` (and a matching XLA-safe Muon variant) avoid `.item()` / Python scalar recompiles via 0-D device tensors under `XLA_NO_SPECIAL_SCALARS=1`. `torch.compile` is off; `torch_xla.compile()` wraps fwd+bwd per micro-step.

MegaCpp production: Megatron-Core specs for NVIDIA, our TPU path unchanged for Google hardware. Model code is one source of truth; the kernel paths fork at the spec boundary.

## Production checklist

- Pick one host framework per lane and adapt, do not chain frameworks inside a single training run.
- On NVIDIA, route through Megatron-Core when you want TE fusion and TP/EP overlap; fall back to FSDP2 + torchtitan patterns when you need PyTorch-native control.
- On TPU, stay on `torch_xla` SPMD with MaxText-derived flags; do not try to port Megatron's comm overlap (XLA does it for you).
- Keep `torch.compile` off on XLA; keep it on for CUDA FSDP2 unless a rank-symmetric collective issue forces eager mode.
- When porting a pattern from framework X to framework Y, copy the numeric constants (bucket sizes, alignment rules) verbatim - they encode months of tuning.
- For mixed MoE + dense runs, the EP dispatcher choice (`alltoall` vs `allgather`) is a framework-level decision, not a model-level one.
- When FSDP2 wraps your model and LoRA is injected later, register explicit grad all-reduce hooks on the LoRA params.
- Pin TPU runtime versions (`torch`, `torch_xla`, `libtpu`, `jax`) per a written TPU setup contract; do not upgrade mid-run.

## Lane summary

| Lane | Wrapper | Optimizer | Compile | Notes |
|---|---|---|---|---|
| NVIDIA pretraining | Megatron-Core DDP | DistributedOptimizer + TP-Muon | TE kernels, no `torch.compile` on hot path | Production for the trunk and dense specialists |
| NVIDIA wide-MoE / LoRA | FSDP2 (`fully_shard`) | AdamW + per-group hooks | `torch.compile` on small-MoE only | LoRA via `ignored_params` or post-wrap hooks |
| TPU v6e pretraining | XLA SPMD ZeRO-3 (our TPU sharding runtime) | DistAdamW + DistMuon | `torch_xla.compile` per micro-step | MaxText-derived flag taxonomy |

A typical CUDA wrap order, kept identical between specialists so the planner can reason about it:

```python
model = build_specialist(cfg)
model = wrap_tp(model, tp_mesh)
model = wrap_pp(model, pp_mesh)          # only if PP > 1
model = apply_cuda_fsdp(model, dp_mesh)  # bottom-up
model = apply_ep(model, ep_mesh)         # MoE specialists only
optimizer = build_megatron_distributed_optimizer(model)
```

## References

- the TPU sharding runtime, the CUDA FSDP2 wrapper, the distributed parallelism module, the TPU flag bundle, the XLA AdamW implementation, the Megatron bridge, the Megatron optimizer wrapper (POC)
- the family-aware Megatron entrypoint, the public authored Mamba spec sample, and the public recurrent spec sample (production)
- the public TPU setup note, an FP8 torchao comparison note, a MaxText feature-merge note
- CHANGELOG entries 2026-03-29 through 2026-04-09 (Megatron + FSDP2 + MaxText pattern transfers)
- [PyTorch FSDP2 documentation - PyTorch]
- [Megatron-Core - NVIDIA]
- [DeepSpeed ZeRO and 3D parallelism - Microsoft]
- [torchtitan - Meta / PyTorch]
- [nanotron - Hugging Face]
- [MaxText - Google]
- [T5X - Google Research]
