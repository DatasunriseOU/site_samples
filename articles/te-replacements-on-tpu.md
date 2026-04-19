---
title: "Transformer Engine replacements on TPU: keeping one model definition across paths"
description: "Transformer Engine is an NVIDIA Hopper and Blackwell story. On TPU v6e it does not exist. This is the layer-spec abstraction and the XLA-friendly substitutes that let one model definition ship across both paths."
date: "2026-04-18"
tags: ["tpu", "v6e", "xla", "transformer-engine", "layer-spec", "fp8"]
---

Transformer Engine is the load-bearing fast path on the GPU path: fused norm+linear, FP8 autocast for the QKV matmul, cuDNN flash attention, fused MoE permute. None of it ports to TPU v6e. The TPU path has a different stack (XLA, Pallas, JAX kernels), a different precision story (no FP8 in deployment today), and a different sharding model. The interesting engineering question is not "how do we get TE on TPU"; it is "how do we keep one model definition that sees TE on the GPU path and a clean XLA-traceable substitute on the TPU path, without forking any block code." This post is about that abstraction and the substitutes we ended up with.

## Why This Matters

Two paths ship the same specialists: CUDA hosts and TPU v6e slices. The training loop, the attention modules, the MoE router, the long-context curriculum, the data pipeline, the tokenizer — all single-source. The only places allowed to diverge are kernel implementations and the precision plan. If the model definition forks, every feature lands twice and the ablations stop being comparable.

The hardest piece of that constraint is the Transformer Engine surface. On the GPU path, TE owns the highest-throughput path for at least four primitives: pre-norm fused into the next matmul, the QKV/MLP column-parallel linears, FP8 attention via `fp8_autocast`, and the fused MoE permute kernel. On the TPU path, none of those exist as TE modules. The substitute has to be (a) numerically equivalent at bf16, (b) traceable by XLA without dynamic shapes, (c) shardable under SPMD without surprise propagation, and (d) selectable at construction time so the same block code lives in both worlds.

## The Shared Layer-Spec Approach

The pivot is the public TE layer-spec sample. It is a smaller adaptation of Megatron-Core's `ModuleSpec` pattern: a plain `dict[str, type | None]` that maps seven block component names — `norm`, `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`, `attention`, `layernorm_mlp` — to TE classes when TE is importable and to `None` otherwise. The mapping mirrors Megatron's TE submodule selection: `linear_qkv` and `linear_fc1` both resolve to `LayerNormLinear` (pre-norm folded into the next GEMM), `linear_proj` and `linear_fc2` are `Linear`, `attention` is `DotProductAttention`, and `layernorm_mlp` is the full-block `LayerNormMLP` fusion.

The contract that makes this useful on TPU is import safety. The module-level `_TE_AVAILABLE` flag is set inside a `try/except` that swallows every failure: missing package, ABI mismatch, broken transitive dependency. When TE is unavailable, calling `te_layer_spec(use_te=True)` returns a dict of `None` values; the caller substitutes natively and never sees an exception. The same import line lives at the top of a block file regardless of platform.

We deliberately did not copy Megatron's full `ModuleSpec` machinery: the blocks (`Block`, `ABlock`, `MBlock`, `EBlock` in the main model runtime module) take module references at `__init__`, not spec objects with `params`/`submodules` fields. A dict is the minimal shim that matches the existing surface and composes with both paths.

On the GPU path, `te_layer_spec(use_te=True)` returns the seven TE classes and `_TEAttentionBlock` assembles them into a fused QKV -> DPA -> output-proj block: pre-norm folded into `LayerNormLinear`, GQA via `num_gqa_groups=num_kv_heads`, `bshd` layout, residual sum at the end.

On the TPU path, where the dict is all `None`, the substitutes are: `F.rms_norm` + `F.linear` for the fused-norm-linear keys (XLA HLO fuses the two into a single GEMM); the Pallas softcap kernel for `attention` exposing the same `(q, k, v, doc_ids, softcap)` surface as TE DPA; a SwiGLU MLP (two `F.linear` + `F.silu(gate) * up`) for `layernorm_mlp`, within single-digit percent of TE's fused kernel; `nn.Linear` + `mark_sharding` for `linear_proj`/`linear_fc2`, letting XLA SPMD pick the collective; and an equal-split bf16 MoE dispatch in the MoE dispatch runtime module (no `.item()`, no data-dependent shapes) in place of the fused `moe_permute_with_probs`/`moe_unpermute` from the public TE permutation sample.

the public Megatron block sample is the block-level adapter: a Megatron `TransformerLayer` wrapper that uses the TE layer spec on the GPU path and degrades silently on TPU via `MEGATRON_AVAILABLE`. Same `(config, layer_idx)` constructor signature, same forward signature.

FP8-on-TPU is the explicit asterisk. We do not ship FP8 on the TPU path today; TE DPA's `fp8_autocast` is the cleanest FP8 attention path on H200, and the TPU equivalent is waiting for libtpu's per-tensor FP8 to mature for our shapes. The TPU path stays bf16 with the same clipping, the same Muon/AdamW split, the same loss target. FP8 is a precision-plan object, not a structural choice — on at construction on the GPU path, no-op on the TPU path.

## How it lands in deployment

The lift-as-is parts: the seven-key dict surface in the public TE layer-spec sample, the `_TEAttentionBlock` assembly, the import-safety pattern. They become the canonical "TE or native" selector and the model definition stops carrying ad-hoc branching anywhere else.

Rewritten on the way in:

1. The native fallback is consolidated. MegaCpp has a dozen places in the main model runtime module where the block code reads `spec[key] is None` and assembles the substitute inline. Production centralizes that into a `NativeLayerSpec` factory with the same key set. The block code calls one factory regardless of platform; the factory returns either TE classes or the XLA-friendly substitutes.
2. The TPU-path substitutes get explicit SPMD partition specs at construction time. Relying on XLA sharding propagation to infer the spec from the surrounding graph has bitten this code repeatedly. Production pins every substitute parameter with `mark_sharding` at the same call site that constructs it.
3. The FP8 plan becomes a precision-plan object rather than a constructor flag. On the GPU path it wraps the relevant linears with `fp8_autocast`; on the TPU path it is bf16 everywhere with a clear log line saying so.
4. the public TE linear-replacement sample's post-hoc rewrite of every `nn.Linear` with `te.Linear` becomes the GPU-path initialization step that runs after model construction and before FSDP wrapping. The exclusion list (`wte`, `wpe`, `lm_head`, `router`, `shared_expert_gate`, `engram`, `mhc`, `ngram_hash`, `structure_emb`, `platform_emb`, `temporal_`, `lora_`) lifts as-is. On the TPU path the rewrite is a no-op.

Dropped: the per-experiment ad-hoc branching for "if TE installed do X else do Y" that grew up across half a dozen modules. It is replaced with the single dict surface.

Moved to a kernel/Pallas path: the attention substitute. The block code calls a single attention adapter; on the GPU path that adapter is TE DPA, on the TPU path it is the Pallas softcap kernel. Same call signature, two backends, one selector.

Becomes a feature flag: `--use_te_block_layers`, `--use_te_all_linears`, and the FP8 precision plan. All three are no-ops on the TPU path and the entry point logs which subset is active so an ablation across paths is unambiguous.

## Ablations and what we kept

The seven-key spec, summarised across paths:

| Spec key | GPU path (TE) | TPU path substitute | Notes |
|----------|---------------|---------------------|-------|
| `norm` + `linear_qkv` | `LayerNormLinear` | `F.rms_norm` + `F.linear` | XLA fuses to a single GEMM |
| `norm` + `linear_fc1` | `LayerNormLinear` | `F.rms_norm` + `F.linear` | Same fusion path |
| `linear_proj`, `linear_fc2` | `te.Linear` | `nn.Linear` + `mark_sharding` | XLA SPMD picks the collective |
| `attention` | `DotProductAttention` | Pallas FA softcap kernel | One adapter, two backends |
| `layernorm_mlp` | `LayerNormMLP` | SwiGLU MLP (two `F.linear`) | Off by default; harder to TP-shard |
| MoE permute | the public TE permutation sample fused | bf16 equal-split path | Same `(idx, gates)` surface |
| FP8 plan | FP8 attn + FP8 experts + bf16 rest | bf16 everywhere | Logged on startup |

The ablation history on this path is mostly about what failed silently. A few items shaped the current form:

The TE in_proj fusion for the Mamba mixer (the public TE input projection sample work) was the cleanest "TE win on GPU, no-op on TPU" win we have. Replacing the Mamba `nn.Linear` in_proj with `TELayerNormColumnParallelLinear` folds the LN into the column-parallel projection, drops one kernel launch, and matches the surrounding TE block precision plan. On the TPU path the same module is a plain `F.rms_norm` + `F.linear`; the XLA fuser does the right thing and the loss curves overlay. Sentinel values inside the wrapper let the block code stay agnostic.

`tp_comm_overlap=True` did not survive contact. The the public Megatron block sample config builder removed it after an audit of the TE extension layer: setting the flag requires a matching `te.initialize_ub(...)` call before model construction, which our GPU-path bring-up path does not do. Leaving the flag on would have produced a latent crash the moment someone ran `--use_megatron_block --megatron_tp --tensor_parallel=2 --sequence_parallel`. We kept the five real fusions (`masked_softmax_fusion`, `persist_layer_norm`, `attention_softmax_in_fp32=False`, `apply_rope_fusion`, `gradient_accumulation_fusion`) which do not depend on user-buffer overlap.

The `LayerNormMLP` full-block fuse is exposed but not the default. Megatron does not use it by default because it makes fc1/fc2 sharding harder for TP; for single-GPU and FSDP-only runs it is the fastest path on the CUDA side. On the TPU path the substitute is the SwiGLU MLP we already had. The block code reads `spec["layernorm_mlp"]`; if non-`None` it uses the full-block fuse, otherwise it uses the substitute.

The "use TE everywhere via post-hoc replacement" path (the public TE linear-replacement sample) survived contact but only with the exclusion list. Replacing embeddings with `te.Linear` clobbers the tied lm_head weight; replacing LoRA adapters wraps an existing Linear and breaks the rank-decomposition; replacing the MoE router silently FP8-quantizes a 1024-wide projection that needs full bf16 precision to keep routing decisions stable. The exclusion list is a load-bearing piece of the contract.

What we tried and did not keep: a "single FP8 plan that works everywhere" that conflated H200 FP8 with a hypothetical TPU FP8. The two are not the same. We pulled them back into separate precision plans, with the TPU plan being explicitly bf16 today and the GPU plan being FP8-attention plus FP8-experts plus bf16-rest. Each plan logs its full configuration on startup; reading "FP8 plan: bf16 everywhere" on a TPU run is the right kind of unsurprising.

## Production checklist

The block-side selector, in one place:

```python
from te_layer_spec import te_layer_spec

spec = te_layer_spec(use_te=True)  # all-None on TPU; TE classes on GPU
norm_qkv = spec["linear_qkv"] or RmsNormLinear     # substitute on TPU
attn     = spec["attention"]   or pallas_fa_softcap_adapter
ln_mlp   = spec["layernorm_mlp"] or SwiGLUMLP
# FP8 is a separate precision-plan object, no-op on TPU.
```

- The seven-key dict from the public TE layer-spec sample is the only place that maps "TE name" to "implementation." Block code reads the dict, never imports `transformer_engine` directly.
- `_TE_AVAILABLE` is a plain bool, set once at import time, with no side effects on the no-TE path. Tests patch it to force the substitute path.
- Every TPU-path substitute parameter is `mark_sharding`-pinned at construction time. No reliance on XLA sharding propagation for the substitute parameters.
- FP8 is a precision plan object, selected at construction time, no-op on the TPU path. The startup log records exactly which plan is active.
- The MoE permute path is the TE fused kernel on the GPU path and the equal-split bf16 path on the TPU path. Both expose the same `(expert_indices, gate_weights)` surface.
- The post-hoc `nn.Linear` -> `te.Linear` rewrite runs before FSDP wrapping and before `torch.compile`, with the documented exclusion list.
- `tp_comm_overlap` stays off until the bring-up path calls `te.initialize_ub(...)`. No exceptions.
- `--use_megatron_block` is opt-in and benchmarked per release. The current measurement on the dense A-block places it well behind the native path on TPU; do not enable it there.
- A loss-curve overlay between the GPU path and the TPU path on the same data shard, same step count, same precision plan is the regression test that fails noisily when the substitute drifts from the TE path.
- The block forward signature is identical across paths; any new TE-backed primitive must come with a documented XLA substitute before landing.

## References

- the public TE layer-spec sample
- the public TE attention sample (GPU-path DPA adapter)
- the public TE block sample (GPU-path TE-native A-block)
- the public TE permutation sample (GPU-path MoE permute)
- the public TE linear-replacement sample (post-hoc Linear rewrite)
- the public TE bridge sample (TE import bridge)
- the public Megatron block sample (Megatron TransformerLayer adapter)
- the TPU attention dispatch layer (dense attention with TPU Pallas/Splash backends)
- the MoE dispatch runtime module (XLA-friendly MoE dispatch)
- the public TE input projection sample (Mamba TE in_proj fusion)
- [Transformer Engine documentation - NVIDIA]
- [Megatron-Core gpt_layer_specs - NVIDIA Megatron-LM]
- [Pallas: a JAX kernel sublanguage - JAX docs]
- [Splash Attention - Google JAX kernels]
