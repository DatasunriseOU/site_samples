---
title: "Transformer Engine on H200 and Blackwell-class GPUs: the bridge we use"
description: "How MegaCpp wires NVIDIA Transformer Engine into the training stack on Hopper and Blackwell, where TE replaces native PyTorch layers, the FP8 interaction, and the fallback path that keeps non-NVIDIA lanes alive."
date: "2026-04-18"
tags: ["transformer-engine", "fp8", "H200", "Blackwell", "nvidia", "training"]
---

Transformer Engine (TE) is one of the biggest performance levers NVIDIA provides on Hopper and Blackwell-class GPUs, and also one of the easier ways to destabilize a multi-host training run. The integration here keeps every TE call optional, late-bound, and behind a per-feature flag, then lifts only the modules that materially improve MFU into a stable deployment layer. This post explains that bridge: what TE buys, where it replaces native PyTorch layers, how FP8 composes on H200 versus smaller Blackwell-class targets, and how MegaCpp keeps a clean path back to a TE-free build for lanes that need it.

## Why we own a TE bridge at all

The hybrid architecture is layout-heavy: dense attention A-blocks, Mamba M-blocks, MoE E-blocks, and DSA-attention layers all share the same depth budget. Each block class has a different FLOP/byte ratio and a different dominant kernel. Running the canonical PyTorch forward leaves a meaningful fraction of achievable MFU on the table; the Megatron-Core TE spec shows the ceiling with a single fused dynamo graph per layer, fused pre-norm + linear, cuDNN flash attention, and a `fp8_autocast` scope that picks per-layer precision. The bridge keeps that ceiling available without making TE a hard dependency on lanes where it is unavailable or incomplete.

The bridge has three jobs: keep the import surface non-fatal when TE is missing, expose TE classes one at a time so we can A/B them against our native blocks, and degrade to the FA3 / native MLP path on the same calling convention. Everything else is downstream of those three.

## The seven modules that make up the bridge

The TE surface is seven small modules, each one targeting a single replacement.

| Module | Replaces | Notes |
|---|---|---|
| `te_bridge.py` | n/a | Import firewall; lazy accessors; one-line per re-exported TE entry point |
| `te_attention.py` | Custom DPA path | Wraps `DotProductAttention` for our `[B, H, T, D]` layout; GQA via `num_gqa_groups`; RoPE applied before the call |
| `te_block.py` | Native A-block | Full TE-native ordering: `LayerNormLinear` + `DotProductAttention` + `Linear` + `LayerNormMLP` |
| `te_layer_spec.py` | Spec dispatcher | `dict[str, type | None]` keyed on Megatron's submodule names; `None` falls back |
| `te_linear_replacement.py` | Walks the model tree | Swaps `nn.Linear -> te.pytorch.Linear`; exclusion list pinned |
| `te_experts.py` | Custom MoE GEMM | Wraps `GroupedLinear` for FP8 expert compute; permute/unpermute via TE primitives |
| `te_permute.py` | Custom permute path | Bridges `moe_permute_with_probs` and friends to our dispatcher |

`te_bridge.py` is the import firewall. Every other file goes through `is_te_available()` and the `te_*` lazy accessors here, so importing any TE-aware module on a TE-less machine never raises. The first call sets a process-wide cache; subsequent calls are free. The bridge re-exports only the TE entry points we actually use; if we need a new primitive, it gets a one-liner here, then a typed wrapper at the call site. This pattern is what makes the rest of the stack import-safe.

`te_attention.py` documents what it cannot do (packed-doc `doc_ids` masking, DSA local-window masks, MoBA top-k routing, soft-cap logits) and falls back to our FA3 path for those. This file locks us into one of the two attention worlds: TE for unconstrained transformer-shape attention on Hopper/Blackwell, FA3 for everything else. `te_block.py`'s value is not the wrappers but the ordering: lifting the entire block into TE classes was the only stable fix for fusion-window breaks where saved-activation layout flipped between eager and compile. `te_linear_replacement.py`'s exclusion list is the interesting part: token embeddings (tied with `lm_head`), MoE routers, mHC score projections, MoD routers, n-gram hash projections, structure embeddings, and LoRA adapters are all skipped because they are either too small to win on FP8, structurally weight-shared, or wrappers around an existing `Linear` that the swap would clobber.

## FP8 composition on H200 and on smaller Blackwell-class targets

FP8 is where the bridge stops being convenient and starts being a contract. On H200 (Hopper), TE's `fp8_autocast` plus DelayedScaling on the standard recipe is the production path; the GEMMs use cuBLASLt FP8 paths and the activations are stored in FP8 with per-tensor scaling. We wrap every TE call site in a `fp8_autocast` context, but only inside the FP8 zone of the model: the BF16 first/last layers and any auxiliary head stay outside. Iter 5 of the recent refactor formalised this for the mHC group loop: a per-group helper probes each layer's installed factory; if any layer in the group sits in the BF16 zone the whole group runs under `nullcontext`, otherwise the group runs under the factory at index 0. That is coarser than per-layer but safer, because double-entering `fp8_autocast` inside `_mhc_group_forward` produced silent precision drift in earlier iterations.

On smaller Blackwell-class targets the picture changes. FP8 paths exist, but the cuBLASLt cooperative algorithms used on H200 are not all available, and `flash-attn` may require target-specific builds. Those systems are treated as compatibility lanes: they must build and run a bf16 forward, but they are not used as the reference for H200 throughput claims. Any FP8 measurements from them are tracked separately from the H200 performance baseline.

```python
# stylised wiring inside one mHC group
with self._mhc_group_fp8_ctx(group_indices):
    out = _mhc_group_forward(
        layers=[self.layers[i] for i in group_indices],
        x=x, **kwargs,
    )
```

## Where the bridge meets parallelism

The TE block plays nicely with Megatron's TP because both speak the same column/row contract; the only adjustment is that the QKV column-parallel split has to be segment-aware so head-grouped attention slices correctly. With sequence parallelism on, `LayerNormLinear` does the all-gather on input and `Linear` does the reduce-scatter on output, so the SP gradient all-reduce on norm/QK-norm parameters has to be installed (we do this in `_install_sp_norm_grad_allreduce` because not every code path of ours goes through Megatron's `finalize_model_grads`). With FSDP2 we wrap TE blocks at the block boundary and let `fully_shard` handle the rest; the only gotcha is `reshard_after_forward=False` for the layers that are immediately re-used by the MTP head.

The MoE path is more interesting. `te_experts.py` plus `te_permute.py` give us FP8 grouped GEMM for the expert bank with permute/unpermute primitives that handle padded and jagged dispatch. On H200 this is a real win, especially with EP > 1 because `GroupedLinear` avoids per-expert kernel launches.

## Design choices that held up

The seven-module bridge layout held up, along with the import-firewall pattern, the `dict[str, type | None]` spec, the per-feature TE flag rather than a global TE switch, the per-group FP8 scope helper, and the rule that auxiliary heads stay native PyTorch. The BF16 first/last zone remains outside `fp8_autocast`, and measurements from smaller Blackwell-class systems remain separated from the H200 baseline.

The design does not keep the early "wrap once at module init" pattern because it broke under TE upgrades, per-layer `fp8_autocast` inside `_mhc_group_forward` because it double-entered the context, or TE for token embeddings and routers because the measured win was negligible relative to correctness risk. The throughput win is selective; the bridge exists because selective adoption is the sustainable shape.

## How the bridge survives a TE upgrade

TE upgrades are a recurring source of regressions because the project moves fast and the public surface is large. Three rules keep the bridge stable across upgrades.

First, every TE entry point we use is re-exported through `te_bridge.py`. When TE renames a class or moves it between submodules, exactly one file changes; the call sites do not. That has paid for itself three times in the last six months: the `moe_permute_with_probs` rename (the previous name was a private `_te_*` symbol), the `LayerNormLinear` move into `te.pytorch`, and the `fp8_autocast` signature change that added a `recipe` keyword.

Second, every TE wrapper carries a parity test against the native PyTorch reference. The `te_attention.py` wrapper has a parity test that builds a small QKV input, runs both paths at fp32, and asserts max-abs and max-rel error against tolerances written next to the math. The `te_block.py` wrapper has a similar test for the full block. When TE bumps and a parity test fails, we know exactly where the divergence is and we can decide whether to bump TE further or pin to the previous version while we investigate.

Third, every TE-using preset has an env-var fallback that disables TE for that preset only. Setting `MEGACPP_DISABLE_TE=1` at launch causes `is_te_available()` to return False even on a TE-installed host, which forces every wrapper to fall back to its native path. That fallback is there so validation and production can drop to the non-TE path quickly if a TE upgrade or driver change regresses correctness.

## Observed H200 impact

Per-block MFU numbers are not published because they are noisy and shape-dependent, but the qualitative picture is stable on H200. Lifting the dense A-block to a TE-native block with `LayerNormLinear`, `DotProductAttention`, and `LayerNormMLP` improved steady-state throughput on the deep hybrid by a low-double-digit percentage. Adding TE FP8 `GroupedLinear` for the MoE expert bank on top of that added a further high-single-digit percentage on EP > 1 configurations. The TE in-proj fusion for the Mamba 2/3 layer (`te.LayerNormLinear` with `normalization='RMSNorm'`) added a smaller but still measurable win on Mamba-heavy presets. None of these gains came from a single switch; they depended on the call-site refactors and parity tests described above.

On smaller Blackwell-class targets the picture is different. TE works, but FP8 paths are not all available, and the wins measured on H200 do not transfer directly. Those lanes are treated as build-and-correctness targets only; their performance numbers are not part of the H200 steady-state matrix.

## Reusable bridge pattern

The selective-import pattern in `te_bridge.py` has held up well enough to reuse for other optional vendor-library integrations, including the Liger fused norm/CE path and the cut-cross-entropy path. The same three rules apply: one re-export file, parity tests against the native reference, and an env-var fallback per preset. That bridge layer is what makes "use TE where it wins, fall back gracefully where it does not" operational instead of aspirational.

## References

- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [Transformer Engine GitHub repository](https://github.com/NVIDIA/TransformerEngine)
- [Megatron Core developer guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [NVIDIA H200 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h200/)
- [Hybrid layout notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md)
