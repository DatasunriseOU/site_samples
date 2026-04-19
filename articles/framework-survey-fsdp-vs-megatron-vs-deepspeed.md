---
title: "Framework Survey: FSDP2 vs Megatron-Core vs DeepSpeed vs Torchtitan vs Nanotron vs MaxText"
description: "Honest comparison of large-scale training frameworks, what each is good at in 2026, and which stacks fit NVIDIA and TPU training lanes."
date: "2026-04-18"
tags: ["fsdp2", "megatron-core", "deepspeed", "torchtitan", "nanotron", "maxtext", "frameworks"]
---

Teams have shipped training code on PyTorch FSDP2, Megatron-Core, torchtitan-shaped patterns, DeepSpeed, and MaxText-inspired TPU setups. This post is a practical map of which framework is good at what in 2026, what the gaps are, and why different hardware lanes often end up using different stacks on purpose.

## Why this matters

Framework choice is a tax on everything that comes after. The wrong choice means you spend months reimplementing sequence-parallel norm grad sync, or diagnosing silent FP8 scope mismatches, or rewriting your data loader when you move from a research mesh to a production mesh. The right choice means the boring distributed-systems machinery is already solved and you can spend your time on architecture. We have bled enough on each of these frameworks to pick deliberately per lane rather than picking one and hoping.

## What teams use today

Three distributed paths commonly show up side by side:

An XLA SPMD ZeRO-3 style TPU runtime walks the model, builds a per-parameter sharding spec over a `("data", "model", "expert")` mesh, and hands those annotations to `xs.mark_sharding`. The XLA compiler inserts the required all-gather, reduce-scatter, and all-to-all collectives during compilation. There is no step-time Python communication scheduler in that model; the collectives are HLO ops.

A CUDA FSDP2 wrapper uses `torch.distributed.fsdp.fully_shard`. It runs after TP wrapping, before compile, and before the optimizer. It has the usual FSDP2 knobs such as `reshard_after_forward` and prefetch control. One practical caveat remains important: if LoRA injection happens after `fully_shard`, the fresh LoRA parameters are not FSDP-managed and need explicit gradient synchronization hooks.

A native PyTorch path composes PP + FSDP2 + EP, optionally with TP as a fourth axis, for CUDA runs that need 3D parallelism. It orders the wrapping `TP -> PP -> FSDP2 -> EP` explicitly, which matches torchtitan-style guidance. This is the heaviest-weight path and it remains useful when PP is the only realistic answer for models that do not fit in a single DP stage.

The integration layer around Megatron-Core exists because teams use it when they want Transformer Engine kernels, TP/EP communication overlap, and a distributed optimizer with bucket coalescing.

On the TPU side, a `torch_xla` path informed by MaxText-style flag taxonomy is common. VMEM limits, continuation-fusion settings, and SparseCore-related flags are treated as tuning surfaces rather than folklore. Borrowing that public taxonomy is much cheaper than rediscovering it from trial and error.

Some teams no longer ship a DeepSpeed path in the main repo after evaluating it early, keeping a few patterns such as bucketed gradient reduction and distributed-optimizer ideas, then moving the implementation onto Megatron-Core or native PyTorch.

## How it lands in practice

The NVIDIA path is narrower. Megatron-Core is often the host on that lane, with model families written against Megatron abstractions directly. There is usually no FSDP2 path in the main NVIDIA training lane; Megatron DDP with the distributed optimizer is the production choice there.

The TPU lane keeps the XLA SPMD path. Mesh shapes are chosen from `("data",)`, `("data", "model")`, `("data", "expert")`, or `("data", "expert", "model")` depending on TP and EP sizes. `torch.compile` is kept off on XLA; the TPU compile contract is per-micro-step `torch_xla.compile()` around forward and backward.

## Ablations and what we kept

**PyTorch FSDP / FSDP2.** FSDP2 is the upstream-blessed path now. Meta's PyTorch team documents it as the recommended direction for new training code, with per-parameter DTensor sharding and a cleaner migration story than FSDP1. For a CUDA lane it is the baseline when you do not need TP or PP: a single-node dense run, or a small MoE where EP=1. The good: it composes with `torch.compile` by default in 2.4+, the mp_policy story (param/reduce/buffer dtypes separately controlled) works, and FSDP2 + FP8 via `torchao.Float8Linear` is ergonomically clean if you wrap in the right order. The bad: it does not give you TP or PP; you compose those yourself. For anything past roughly one node of dense training, teams often end up on Megatron-Core.

**Megatron-Core.** The ceiling for NVIDIA training. Transformer Engine integration is where MFU comes from on Hopper and Blackwell: `TELayerNormColumnParallelLinear` + `TERowParallelLinear` + `TEDotProductAttention` with userbuffer comm-overlap, fused RoPE, fused masked softmax, and a working `fp8_autocast` recipe. TP and EP communication overlap is not theoretical. The distributed optimizer buckets gradients by parameter count with NVLink/NVSwitch-aware padding, coalesces reduce-scatters through `_coalescing_manager`, and overlaps param gather with compute. The cost: it is NVIDIA-shaped. `TransformerConfig` is a Procrustean bed for hybrid architectures (see our Megatron porting post), and features without analogues - expert_choice, null_rho routing, attn_softcap, mHC residual streams - either need adapter work or do not port.

**DeepSpeed.** ZeRO-1/2/3, 3D parallelism, ZeRO-Infinity, and MoE primitives in one package. It is genuinely productive for teams that inherited it early, and the ZeRO offload paths are still competitive for memory-bound experiments. Where it is not our first pick in 2026: the moment you want TE fused kernels and FP8, you are gravitating back to Megatron-Core; the moment you want the cleanest upstream-PyTorch integration, you are on FSDP2. DeepSpeed sits in between and does not decisively win either axis for our workloads. We respect it, we are not running it.

**torchtitan.** Not a framework so much as a reference implementation of how to do FSDP2 + TP + PP + Float8 + compile in native PyTorch, without a framework wrapper. Use torchtitan as a pattern source, not as a dependency.

**nanotron.** Minimalist 3D-parallel PyTorch library from Hugging Face. It is readable and teaches the abstractions well. It is useful as a conceptual reference, but not as the main production training lane.

**MaxText.** The JAX/TPU flagship. Production-grade on TPU v5e/v5p/v6e/v7, written against JAX and XLA from the start, and the source of most of the real-world libtpu tuning knowledge in the open-source world. Our TPU flag bundle is a descendant, which is the right level of dependency: we inherit the flag taxonomy and the value ranges, we do not take on JAX. Trying to run MaxText directly would have meant rewriting our model in JAX, which is a non-starter for a codebase that also runs on CUDA.

**T5X.** Mature JAX research framework, mostly seq2seq and Google-research-shaped workflows. It is not where frontier decoder-only LLMs train in 2026 - MaxText has absorbed that role on the JAX side. We have not used it and do not plan to.

The ablations that stuck are mostly cross-framework pattern transfer: bucket-size defaults inspired by Megatron guidance, overlap and prefetch ideas influenced by torchtitan, and TPU memory controls informed by MaxText. The bigger lesson is that no single framework is complete for a hybrid architecture.

## Lane assignments

Here is a representative split in 2026:

NVIDIA pre-training lane: Megatron-Core is the host, TE kernels do the compute, and Megatron DDP with the distributed optimizer does the communication. A native-PyTorch path remains available for runs where you deliberately want FSDP2 + PP + EP control, usually for architecture debugging where you need finer-grained control than Megatron gives.

TPU pre-training lane: `torch_xla` with SPMD ZeRO-3 over a mesh that scales to `("data", "expert", "model")`. MaxText-informed flag taxonomy is applied before `torch_xla` import. XLA-safe optimizer variants avoid Python scalar recompiles via device tensors. `torch.compile` is off; `torch_xla.compile()` wraps forward and backward per micro-step.

In a mixed-hardware organization, Megatron-Core specs often stay on NVIDIA while an XLA path serves Google hardware. Model code can remain one source of truth even when kernel paths fork at the spec boundary.

## Production checklist

- Pick one host framework per lane and adapt, do not chain frameworks inside a single training run.
- On NVIDIA, route through Megatron-Core when you want TE fusion and TP/EP overlap; fall back to FSDP2 + torchtitan patterns when you need PyTorch-native control.
- On TPU, stay on `torch_xla` SPMD with MaxText-derived flags; do not try to port Megatron's comm overlap (XLA does it for you).
- Keep `torch.compile` off on XLA; keep it on for CUDA FSDP2 unless a rank-symmetric collective issue forces eager mode.
- When porting a pattern from framework X to framework Y, copy the numeric constants (bucket sizes, alignment rules) verbatim - they encode months of tuning.
- For mixed MoE + dense runs, the EP dispatcher choice (`alltoall` vs `allgather`) is a framework-level decision, not a model-level one.
- When FSDP2 wraps your model and LoRA is injected later, register explicit grad all-reduce hooks on the LoRA params.
- Pin TPU runtime versions per a written TPU setup contract; do not upgrade mid-run.

## Lane summary

| Lane | Wrapper | Optimizer | Compile | Notes |
|---|---|---|---|---|
| NVIDIA pretraining | Megatron-Core DDP | DistributedOptimizer + TP-Muon | TE kernels, no `torch.compile` on hot path | Production for the trunk and dense specialists |
| NVIDIA wide-MoE / LoRA | FSDP2 (`fully_shard`) | AdamW + per-group hooks | `torch.compile` on small-MoE only | LoRA via `ignored_params` or post-wrap hooks |
| TPU v6e pretraining | XLA SPMD ZeRO-3 | DistAdamW + DistMuon | `torch_xla.compile` per micro-step | MaxText-derived flag taxonomy |

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

- https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- https://docs.pytorch.org/docs/stable/notes/ddp.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- https://www.deepspeed.ai/tutorials/zero/
- https://github.com/pytorch/torchtitan
- https://github.com/huggingface/nanotron
- https://github.com/AI-Hypercomputer/maxtext
- https://github.com/google-research/t5x
