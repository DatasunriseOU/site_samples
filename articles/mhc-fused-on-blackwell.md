---
title: "Multi-Head Cross fused on Blackwell: from reference einsum to Triton"
description: "How the MegaCpp Multi-Head Cross branch mixer went from a readable PyTorch reference to a fused Triton path on Hopper and Blackwell, and how it lands in deployment through a narrow feature contract."
date: "2026-04-18"
tags: ["mhc", "hyperconnections", "triton", "Blackwell", "H200", "fused-kernels"]
---

Multi-Head Cross, which we call mHC, is the part of the MegaCpp hybrid recipe that mixes multiple residual streams between blocks. The algebra is four `einsum`s and a Sinkhorn normalisation; the pain is that those `einsum`s run per block for every layer, on every token, across Hopper and Blackwell alike, and the reference PyTorch path is too launch-heavy to ignore at our depth. This post is the engineering story of collapsing the mHC reference into a fused Triton path, keeping a safe fallback, and shipping the result as a narrow feature contract in the MegaCpp deployment stack.

## Why MegaCpp cares about this

The dense baseline uses 4-stream hyper-connections in the cross-layer HyperConnections sense, with mHC doing the branch-mixing. Per-block the math is cheap but launch-dominated: a pre-mix (`bn,btnd->btd`), the block body, a residual mix-back (`bnm,btmd->btnd`), and an add. At a depth-52 dense preset that is roughly 200 tiny `einsum` launches per forward, and an early benchmark on H200 showed mHC overhead as the dominant non-attention cost once Mamba-3 and MoE were compiled out of the way. The reference implementations we looked at (lucidrains/hyper-connections' batch-dim packing, MaxText's JAX path with `(B, S, K, D)` layout, AndreSlavescu/mHC.cu's CUDA extension) made different trade-offs; we picked a fused-but-narrow Triton path that keeps the dynamic dispatch simple.

## What we built in the MegaCpp training stack

The starting point is `mhc.py`. `ManifoldBranchMixer` is the reference branch mixer: project each pooled branch representation through a small hidden layer, score it, build a dense `(N, N)` affinity matrix via `bmm` on low-dim keys, and run Sinkhorn normalisation in fp32 to project onto the Birkhoff polytope. For N=2 we detect Sinkhorn's degenerate doubly-stochastic case (which always collapses to uniform) and route through a direct softmax instead. `blend_alpha` blends the learned weights with uniform; we use this to prevent slot-order bias from creeping in when the early training signal is weak. That module stays the reference, and it is what the MegaCpp training stack runs by default.

The fused path lives in `mhc_fused.py`. The scope is intentionally narrow: only the *static 4-stream cross-layer HC path* is wired through Triton, plus a pooled dynamic variant. Everything else - variable N, dynamic per-token weights in MaxText's token-wise mode - stays on the native PyTorch path. The reason is that the hot surface is four named kernels - `fused_stream_mix`, `fused_branch_input`, `fused_add_residual`, `fused_mix_add` - with shape guarantees tight enough to write as direct Triton kernels, while the other paths would each need a separate kernel family.

The forward math in its Torch form is four `einsum`s:

- `fused_branch_input_torch`: `branch_input = einsum('bn,btnd->btd', H_pre, H)`.
- `fused_stream_mix_torch`: the pre-mix plus `H_residual = einsum('bnm,btmd->btnd', H_res, H)` in one call so the backward can share `H_pre`, `H_res`, `H`.
- `fused_add_residual_torch`: `H_new = H_residual + H_post[:, None, :, None] * branch_output[:, :, None, :]`.
- `fused_mix_add_torch`: the residual mix-back fused with the add, which is what actually avoids holding `H_residual` across the block body on the critical path.

Each reference function has an `autograd.Function` partner (`_FusedBranchInput`, `_FusedStreamMix`, `_FusedAddResidual`, `_FusedMixAdd`) whose backward is a hand-written `einsum` chain. The Triton fast path is forward-only; the backward falls back to the explicit PyTorch formulas. That matches the pattern we use elsewhere in the MegaCpp training stack for fused CUDA primitives where the custom backward is not yet worth the complexity.

The fused-mix-add primitive, inlined for reference:

```python
# fused_mix_add_torch (reference)
H_residual = torch.einsum('bnm,btmd->btnd', H_res, H)
H_new      = H_residual + H_post[:, None, :, None] * branch_output[:, :, None, :]
```

Backend resolution at a glance:

| Check                         | Fast path?             |
|-------------------------------|------------------------|
| `is_cuda` and N == 4          | Triton                 |
| N != 4 or token-wise dynamic  | torch (reference)      |
| non-CUDA / shape drift        | torch + one-time warn  |
| fp8 autocast inside group     | group enters once      |
| Sinkhorn (all shapes)         | fp32 with eps clamp    |

The Triton forward kernels themselves are grid-launched with a `(B*T, ceil(D / BLOCK_DIM))` shape. The one that is not a tile kernel is `_fused_dynamic_weights_kernel`, which handles the pooled dynamic variant. That kernel reads the flattened `(B, 4*D)` input, does an on-the-fly RMS and a fused projection through the combined `phi` matrix, applies per-group biases, runs a small Sinkhorn on the `(B, 4, 4)` output, and writes the three weight tensors (`H_pre`, `H_post`, `H_res`). The kernel is specialised to the 24-column output, so we hard-unroll accumulators `acc0`..`acc23` and bias loads `b0`..`b23`. It is ugly in Python form, which is exactly why we would not write it this way for generic N; for the one shape that matters in deployment it compiles to a clean single-kernel launch and dodges the otherwise-expensive 24-way reduction.

Backend resolution is a small state machine in `_resolve_backend` and `_resolve_dynamic_backend`. Requested backend (`auto`, `torch`, `triton`) composes with a module-level cached env var and the Triton-availability probe `_can_use_triton(H)`. `_can_use_triton` is pointed: it only returns true for `is_cuda`, `ndim==4`, `shape[2]==4` (static 4-stream), and dtype in `{fp16, bf16, fp32}`. Anything else - including shape drift from a DTensor/TP split - falls back to the torch path with a one-time warning. The env-var check is done at module import time because `os.environ.get` is not dynamo-traceable; `_refresh_env_backends()` is a test hook for rewriting the cached value.

Numerical guards are the part that took most of our iteration time. Sinkhorn is the obvious one: we run it in fp32 with an `eps` clamp on the row/column sums. `raw_matrix.abs() + eps` before Sinkhorn in the dynamic path keeps the matrix positive, and the N=2 short-circuit in the static mixer exists specifically because Sinkhorn's 2x2 case converges to uniform in a way the Triton kernel cannot distinguish from the learned signal. For the forward kernel we use tensor-resident `torch.minimum` and `amax` guards rather than host-syncing Python scalars, which is the same XLA-compat discipline we use in `qk_clip.py`. `_warn_once` is deliberately skipped inside a compiled region - mutating a Python set and emitting a warning from a traced graph would force a dynamo break on every layer, and we care more about not breaking the graph than about the diagnostic.

`kernels.py` and `triton_kernels.py` are the siblings for other fused paths (fused RoPE for Q and K, row-gather primitives for sparse FA3 packers, Liger / CCE fused cross-entropy backends). We share `_calculate_settings` (next power of two, warp count by block size) and the `@torch.compiler.disable` discipline on Triton wrappers so dynamo does not try to trace into kernel launches. Importantly, the fused RoPE and the fused-CE paths are also auto-resolved: `KernelBackend` = `current | liger | cce | triton`, and `triton` is an alias that resolves to `liger` when available and falls back to `current` otherwise, so mHC is not the only place where "prefer fused, fall back safely" is the default.

## How it lands in MegaCpp

The MegaCpp deployment stack is built on Megatron-Core. The relevant file is the public MHC config sample. Its role is a fail-closed config surface for mHC as a feature contract: it does not re-implement the mixer, it defines a frozen `MHCConfig` dataclass that downstream code reads. `layer_indices` (parsed with the same helper we use for Engram), `n_streams` (must be > 1), `sinkhorn_iters` (> 0), `temperature`, `epsilon`, `blend_alpha`, `dynamic`, `dynamic_mode` (`maxtext` or `pooled`), `fused_ops`, `recompute_group_size`. The classmethod builder takes training-style kwargs and validates them.

The key deployment-stack decision is that mHC does not have a Megatron-native emitter yet. In the public Megatron-args sample the plan builder explicitly notes that mHC remains custom rather than using a Megatron-native emitter. That is not a bug; it is a deliberate seam. The mixer code itself remains `mhc.py` and `mhc_fused.py`; the Megatron-facing side imports that library surface rather than vendoring or rewriting it. The config surface above is the only thing the deployment stack owns. That gives us exactly one place to validate mHC parameters and exactly one place that wires them into Megatron's block layout.

The fused-ops toggle (`fused_ops=True`) is the deployment flag. When it is on, the mixer resolves the Triton backend and uses `_FusedBranchInput` / `_FusedStreamMix` / `_FusedMixAdd` during forward. When it is off (or when the tensor shape does not match `_can_use_triton`'s contract), we fall back to `fused_*_torch`. Our deployment presets keep `fused_ops=True` on H200, where the wall-clock win is measurable; on GB10 the mixer is launch-small enough that the Triton fast path still helps but is not load-bearing. `recompute_group_size` is the knob we use in the MegaCpp production-codebase block loop to amortise recompute cost over N consecutive mHC layers, not per-layer.

One piece that moved from the MegaCpp training stack to the deployment stack is the fp8 autocast scope. On the non-mHC block loop the per-block `fp8_autocast` scope is straightforward. The mHC path is different because `_mhc_group_forward` processes multiple layers as a group, and naively entering `fp8_autocast` per layer would double-enter inside the group. The fix was a `GPT._mhc_group_fp8_ctx(group_indices)` helper and three call sites where the group loop itself enters the fp8 context once; individual layer-level enters are a no-op when the helper flags an already-entered group.

## Ablations and what we kept

The honest ablation history is preserved in public benchmark notes and code receipts. The mHC paper claims a ~6.7% speedup at our scale; our first measurement at the deep dense scale showed mHC (n=4) adding 32% step time because 4x the memory bandwidth from carrying four streams was the dominant cost. That measurement drove the implementation survey: `AndreSlavescu/mHC.cu` (pure CUDA, 1.3x fwd+bwd speedup, 7-15x fwd alone), lucidrains/hyper-connections (batch-dim packing), MaxText's `(B, S, K, D)` JAX layout, and the `WithNucleusAI/mHC-triton` kernels we eventually adapted.

We built the CUDA extension from mHC.cu on H200 with the CUDA 13.2 toolkit. It measured at ~2.6 ms per layer forward, 9.2 ms forward+backward; at depth 52 that is ~480 ms of overhead, a bit under a quarter of the step. That was not a fast enough win to justify a second build toolchain in the image, so we kept it as a reference point, took the forward-only Triton path from `WithNucleusAI/mHC-triton`, and fused the residual-update step with the backward written in explicit PyTorch. The explicit backward is where we accept the slowdown the reference CUDA extension avoids; until the bandwidth side of the problem is fixed, the fused backward is not where the overhead lives.

Dynamic mHC stayed on the PyTorch path intentionally. MegaCpp's `ManifoldBranchMixer` already handles the general N case with the N=2 short-circuit and the `blend_alpha` ramp. The Triton dynamic weights kernel covers only the pooled donor-style formulation with combined `phi`, and only for `in_dim = phi.shape[0]` and output dim 24. We keep both because the pooled path is the right default for training, and the token-wise MaxText-style path is the one we fall back to for experiments that need per-token routing.

The `muon` / deep-hybrid / mHC interaction is in the other half of the ablation story. At a depth-52 hybrid preset with mHC, raw Muon diverged to NaN at step 1-6. The fix was split-QKV Muon plus the hybrid interleaving; the mHC layer count stayed constant at 4 streams. That receipt is also the one that established our deployment mHC defaults: 4 streams, first 7 layers as mHC layers. The earlier-numbered alternative we tried without mHC (`_no_mhc_recovery_v1`) was explicitly carved out as a bisect alias so we could distinguish "the mHC contract is wrong" from "something else broke"; it is not a silent fallback.

## Production checklist

- mHC is library code. the MegaCpp deployment stack owns the config surface (the public MHC config sample), while the implementation lives in `mhc.py` and `mhc_fused.py`.
- `fused_ops=True` is the default on H200. GB10 keeps it on for parity; the win is smaller.
- The Triton fast path is strictly the static 4-stream case. `_can_use_triton` is the single gate; shape drift falls back to `fused_*_torch` with a one-time warning.
- Backend env vars are cached at import time and refreshed only through `_refresh_env_backends()` in tests. Reading `os.environ` on the critical path breaks dynamo tracing.
- The backward path is the explicit PyTorch chain. Do not hand-roll a Triton backward without a recipe that shows it dominates step time.
- fp8 autocast scope enters at the mHC group level, not per layer. Adding a per-layer enter inside `_mhc_group_forward` will double-enter.
- Dynamic mHC stays on the pooled path by default. Token-wise dynamic is experimental and does not have a Triton fast path.
- `blend_alpha` and the N=2 softmax short-circuit are not tunables at inference; they are part of the training contract.
- Sinkhorn runs in fp32 with the `eps` clamp. Do not promote or demote the Sinkhorn precision without a new receipt.

## References

- `mhc.py`, `mhc_fused.py`, `kernels.py`, `triton_kernels.py` in the MegaCpp training stack.
- the public MHC config sample and the public Megatron-args sample in the MegaCpp deployment stack.
- [HyperConnections - Zhu et al., 2024].
- [MaxText HyperConnections (JAX) - Google].
- [lucidrains/hyper-connections - GitHub reference implementation].
- [AndreSlavescu/mHC.cu - CUDA reference implementation].
- [WithNucleusAI/mHC-triton - Triton reference kernels, MIT license].
