---
title: "FSDP2 on the XLA Backend: Making ZeRO-3 Compile Cleanly on TPU v6e"
description: "Field notes on running FSDP2-shaped sharding through XLA SPMD on TPU v6e: the reshard policy that compiles, the recompile traps unique to the XLA backend, and what we ship in production."
date: "2026-04-18"
tags: ["tpu", "xla", "spmd", "fsdp2", "zero-3"]
---

On CUDA, FSDP2 is a wrapper graph: `fully_shard` rewrites a module group into an FSDPParamGroup with hooks that issue all-gather and reduce-scatter. On XLA that machinery is the wrong abstraction. XLA SPMD already inserts collectives during HLO compilation; what we owe it is a sharding spec per parameter and the promise the spec will not change between iterations. This post is how we get ZeRO-3-shaped memory savings on TPU v6e while staying inside the SPMD contract, and what bites you when FSDP2-on-XLA is the lane.

## Why MegaCpp cares about this

TPU v6e has 31.25 GiB HBM per chip. A 4.7B MoE preset with full fp32 optimizer state (AdamW moments plus bf16 master) does not fit on one chip without parameter sharding. TP alone covers only matmul weights; embeddings, LM head, indexer projections, structure extras, and auxiliary banks have to be sharded somewhere. ZeRO-3 is the right shape; on multi-host slices (v6e-16, v6e-32) it is the only way the moments fit. On XLA we cannot wrap modules, so we convince the SPMD compiler to materialise the same memory pattern through `mark_sharding` annotations on a sensible mesh.

The shipped MegaCpp ensemble has both backends: H200 nodes use FSDP2 wrappers from `torch.distributed.fsdp`; TPU v6e slices use the XLA SPMD ZeRO-3 path described here. Two implementations behind one operational contract: same parameter classifier, same fp32 reduce promise, same explicitly-replicated set. Everything else differs.

## What we built in the POC
The XLA path lives in the public FSDP sharding sample, roughly 400 lines, intentionally not symmetric with the CUDA the CUDA FSDP companion sample wrapper. The single entrypoint `apply_fsdp_sharding(model, mesh, tp_degree, dp_degree, ep_degree)` walks `named_parameters()` once and calls `xs.mark_sharding(param, mesh, spec)` once per parameter. After that, training is plain `torch_xla.compile` plus SPMD propagation; no per-block hooks, no Python collective-injection layer. The spec resolver uses two helpers: - `_get_tp_sharding_spec(name, param, tp_degree)` returns the TP spec from a name-pattern classifier: fused QKV and MLP gate/up are column-parallel `("model", None)`; `attn.c_proj` and `mlp.c_proj` are row-parallel `(None, "model")`; MLA up-projections are head-sharded along dim 0; MLA down-projections (`w_dq`, `w_dkv`, `w_dqkv`) are explicitly replicated to stop XLA inferring a sharding that happens to divide the LoRA rank by `tp_degree`. LoRA `lora_A` / `lora_B` follow the base layer's direction via `_classify_lora_param`. - `_pick_fsdp_shard_dim(name, param, tp_spec)` picks the dimension to add `"data"`. For 2D TP-split weights it shards the *other* dim, producing `("model", "data")` or `("data", "model")`. For 1D and small tensors it replicates; all-gather overhead is not worth the bytes. Three behaviours fall out: 1. Pure DP-sharded params get `("data", None, ...)`. SPMD inserts all-gather on read, reduce-scatter on grad. 2. 2D (TP + DP) sharded params: matmul weights are 2D-tiled across `("data", "model")`. Optimizer state is `1/(dp_degree * tp_degree)` the size. 3. Replicated by intent: small biases, norm weights, MLA latents, and the host-resident n-gram hash table get an explicit replicated annotation.


## SPMD vs FSDP semantics

The CUDA FSDP2 path runs in the eager Python graph. `fully_shard` installs hooks; at every block boundary Python decides whether to re-all-gather, when to free unsharded shards, when to overlap. The knobs (`reshard_after_forward`, `prefetch_limit`, `MixedPrecisionPolicy`) are runtime decisions evaluated step by step. Getting them wrong costs OOM and slow steps.

The XLA SPMD path runs once, at compile time. `mark_sharding` is a hint to the GSPMD partitioner, which inserts the collectives inside the HLO graph subject to its own scheduling. There is no `reshard_after_forward` and no `prefetch_limit`; resharding and prefetch are compiler decisions. The knobs (`xla_jf_vmem_memory_space_assignment`, the MSA flags) are compile-time and baked into the cached HLO. Getting them wrong costs a graph that does not fit, an MSA crash, or a recompile cliff.

FSDP2-on-XLA is not a port of FSDP2; it is the same memory shape achieved with a different mechanism. The lift from CUDA is the parameter-shape policy; from FSDP2 itself, essentially nothing.

## DTensor compatibility plumbing

Some model code was written against FSDP2's DTensor wrappers. On XLA we do not get DTensor on parameters the same way, but a few modules (the embedding lookup, a couple of fused-CE paths) had hard-coded `to_local()` calls or relied on `placements`. the DTensor utility sample is the compatibility layer that keeps them working under both backends. It provides `is_dtensor_like`, `to_local_if_dtensor`, `maybe_replicate_to_match`, and a `_dtensor_safe_embedding_impl` that does the right thing whether the weight is a real DTensor, an SPMD-sharded XLA tensor, or a plain CPU tensor. The trace diagnostics came out of debugging a wedge where the outer Parameter carried mesh metadata while the dispatch value was plain Tensor-backed; the resolver prefers the backing data carrier and falls back to the wrapper.

## The reshard policy that actually compiles cleanly

On the XLA side, the "reshard policy" is the parameter classifier plus the mesh, and there are exactly three things that have to be true for it to compile cleanly on v6e:

1. **Every annotated dimension must be evenly divisible by its mesh axis.** The `_pick_fsdp_shard_dim` helper checks `param.shape[shard_dim] % dp_degree`, falls back to the other dimension for 2D tensors, and replicates if neither dim works. The bug we used to ship had it silently sharding to the nearest power of two; the partitioner accepted it and inserted a padded all-gather that drowned the step time.
2. **No two FSDP additions on a TP-sharded dim.** This is what the explicit fall-through in the resolver guards: if both `tp_spec[shard_dim] is not None` and we tried to overwrite it with `"data"`, we drop the FSDP shard and re-emit the original TP spec. The XLA compiler will accept a double-shard via cross-mesh ops, but the propagation downstream becomes unpredictable and you get phantom replication.
3. **MLA down-projections must be explicitly replicated.** This is the single most important footgun: `q_lora_rank` and `kv_lora_rank` happen to divide by 8 in our recipes, so without an explicit spec the partitioner infers a sharding that breaks the latent contract. The resolver returns an all-`None` tuple for `.w_dq.weight`, `.w_dkv.weight`, and `.w_dqkv.weight` to slam the door on that.

The resulting summary log is the artefact we read first after every preset bringup: how many parameters ended up in each bucket (DP-only, 2D, TP/EP-only, replicated), and what fraction of element bytes are actually sharded. On a NAM-shaped MoE preset that number is around 95%. On a dense long-context preset it is closer to 88%, dragged down by replicated norms and small biases that we let stay where they are.

## Recompilation traps unique to FSDP2-on-XLA

The traps are not the same as on CUDA. We have collected four classes that bite specifically when SPMD ZeRO-3 is involved.

**Mesh axis drift across runs.** SPMD's compile cache is keyed on the mesh shape and the spec graph. Adding or removing the `"expert"` axis between launches invalidates every cached graph for the model, which on a v6e-16 slice is a 232-second first step. The fix is to pick the mesh up front from the runtime config and never reshape it inside auto-fit; the auto-fit MoE mesh search now serializes the chosen mesh and refuses to alternate inside one cached run.

**LoRA grad-norm initialisation.** the main training script originally initialised the LoRA grad-norm capture buffer inside the GPU branch. On the XLA SPMD branch that produced a `NameError` the moment LoRA was enabled. We pre-initialise `_lora_grad_norm_captured = None` before entering either branch. Cheap, one-line, regression-tested.

**`.item()` host syncs.** Anything that calls `.item()` inside the compiled region forces a host-device round trip and either falls out of the cached graph or stalls the pipeline. The MoE router, MoD capacity logic, MTP depth loop, and Mamba2 padding dispatch all had to learn `_safe_item` and `_is_xla_device` patterns. The Megatron `clip_coef.item()` in `distributed_clip_grad_norm_()` got dropped entirely because clipping by 1.0 is a no-op the compiler elides anyway.

**Muon under SPMD with TP > 1.** This one took a week to root-cause. Muon's Newton-Schulz iteration uses `torch.stack` on a list of parameter chunks, and on SPMD-sharded parameters the stack lost the sharding annotation, producing NaN from step 1 on v6e-16 / v6e-32. The fix routes column- and row-parallel projection weights to AdamW when running under XLA with `tp_degree > 1`, and uses `expm1`-style numerics inside Muon. The classifier in the main model runtime module now takes a `spmd_tp_degree` parameter explicitly because XLA SPMD does not set the per-config `tp_degree` field that the CUDA path uses.

## How it lands in production

The production path keeps the `apply_fsdp_sharding` resolver almost as-is. The classifier, the dimension picker, and the validator move over verbatim. What changes:

- The mesh is constructed once by the recipe (the public Megatron-args recipe sample and the launch helpers) instead of being inferred per launch. This kills the auto-fit-induced cache invalidations.
- The MLA-replication rules are lifted into a small whitelist consumed by both the model builder (the public GPT builder sample) and the SPMD annotator, so the latent-contract guarantees are stated in one place rather than reimplemented.
- `validate_sharding` runs after the mesh is set and emits a structured event the launch script reads. If the replicated byte fraction is above the per-recipe ceiling, the launcher refuses to enter the long compile.
- The DTensor compatibility layer ships as the DTensor utility sample essentially unchanged; it is small and shared between backends.

What we drop: the `MEGACPP_FSDP2_NO_RESHARD_AFTER_FWD` and `MEGACPP_FSDP2_PREFETCH_LIMIT` env knobs do not exist on the XLA path and never will. They are CUDA-only tuning surfaces. Not having them on TPU is a feature; SPMD owns those decisions.

What becomes a feature flag: the explicit `expert` axis is a per-recipe choice, gated by whether the preset is MoE. The classifier branches on `has_expert_axis` already, but the launcher writes that into the recipe rather than computing it.

## Ablations and what we kept

The ablation history that shaped the current XLA path, drawn from CHANGELOG entries:

- **Mesh shapes for MoE on v6e-16.** EP=4, TP=2, DP=2 with a 3D HybridMesh (2, 4, 2) was the only configuration that compiled and ran a 4.1B MoE preset cleanly. EP=8, TP=2 went OOM at the 22 GiB graph mark. The classifier now writes the EP shard onto dim 0 of the 3D MoE expert tensor and lets DP+TP land on the other two axes.
- **Muon NaN on TPU with TP > 1.** Resolved by routing affected weights to AdamW under XLA SPMD with `tp_degree > 1` and switching to `expm1` numerics in Muon. Verified clean across v6e-16 and v6e-32 over 20 iterations after the fix.
- **libtpu nightly regressions.** A v0.0.37 build broke `--xla_disable_hlo_passes` (the recognised flag became `--xla_jf_vmem_memory_space_assignment=false`) and produced systematic NaN on a NAM-shaped dense preset across multiple torch builds. We pinned the libtpu version, added `--no_compile_optimizer` as an escape hatch (uses `xm.mark_step()` instead of `torch_xla.compile()` around the optimizer step), and now refuse to roll a libtpu nightly without a clean canary.
- **Auxiliary modules.** N-gram hash, structure embeddings, platform embeddings, TreeFFN, and the mHC layers all earned dedicated handling: the n-gram hash table stays on CPU and is explicitly replicated; the rest take the default classifier path. None of them justified custom mesh axes.
- **Validation accounting.** The replicated-bytes report originally double-counted parameters that were excluded from training. It now counts only eligible validated parameter bytes, and treats explicit `REPLICATED` annotations as unsharded so the number matches operational reality.
- **First-step compile time.** A v6e-16 NAM-shaped dense preset hits 232 s on the first compile, drops to 77 s on the optimizer-variant compile, and 24 s steady. We log all three so a regression in any of them is visible.

## Production checklist

- Pin libtpu and `torch_xla` builds; canary every nightly before promotion.
- Construct the SPMD mesh once at recipe boot; never reshape inside auto-fit.
- Run `validate_sharding(model, tp_degree)` after annotation; gate launch on the replicated-byte budget.
- Keep MLA down-projections (`w_dq`, `w_dkv`, `w_dqkv`) on the explicit-replicate whitelist.
- Route Muon-driven 2D matmul weights to AdamW under SPMD when `tp_degree > 1`.
- Forbid `.item()` and `nonzero()` inside compiled regions; use `_safe_item` and pre-allocated scalar buffers.
- Keep the n-gram hash table on host with the device-skip guard intact.
- Log first-step, second-step, and steady-state compile times every run; alert on regressions.
- Treat `reshard_after_forward` and `prefetch_limit` as CUDA-only knobs; do not surface them on the XLA path.
- Use the same parameter classifier across CUDA FSDP2 and XLA SPMD; diverge only where the mechanism forces it.

## XLA SPMD vs CUDA FSDP2 contract

| Concern | CUDA FSDP2 | XLA SPMD ZeRO-3 |
|---|---|---|
| Where collectives live | Python hooks at block boundaries | HLO ops inserted by GSPMD |
| Knob surface | runtime (`reshard_after_forward`, `prefetch_limit`) | compile-time (MSA flags, VMEM limits) |
| Param spec API | `fully_shard(module)` | `xs.mark_sharding(param, mesh, spec)` |
| Failure mode | OOM, slow step | recompile cliff, MSA crash |
| LoRA | `ignored_params` or post-wrap hooks | `_classify_lora_param` follows base layer |

The `apply_fsdp_sharding` entrypoint is one walk over `named_parameters()`:

```python
def apply_fsdp_sharding(model, mesh, tp_degree, dp_degree, ep_degree):
    for name, p in model.named_parameters():
        tp_spec = _get_tp_sharding_spec(name, p, tp_degree)
        spec = _pick_fsdp_shard_dim(name, p, tp_spec)
        xs.mark_sharding(p, mesh, spec)
    validate_sharding(model, tp_degree)
```

## References

- the public FSDP sharding sample, the DTensor utility sample
- the public GPT builder sample, the public Megatron-args sample, the public NAM56R launch sample
- the public engineering changelog entries on Muon NaN under SPMD, libtpu 0.37 regressions, NAM-shaped FSDP profiling, and the v6e-16 / v6e-32 mesh search.
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs — Xu et al., 2021]
- [PyTorch/XLA SPMD documentation — official `torch_xla` docs]
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — Rajbhandari et al., SC20]
