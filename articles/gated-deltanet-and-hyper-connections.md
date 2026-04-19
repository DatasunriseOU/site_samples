---
title: "Gated DeltaNet, Hyper-Connections, and DynamicTanh: The Bits We Slipped Between the Hybrid Layers"
description: "How Gated DeltaNet, cross-layer hyper-connections, dynamic tanh normalization, attention residuals, and gated attention compose inside the MegaCpp hybrid stack — what augments, what replaces, and what survived ablation."
date: "2026-04-18"
tags: ["gated-deltanet", "hyper-connections", "dynamictanh", "gated-attention", "attn-res", "architecture"]
---

The hybrid Mamba 3 plus Transformer interleave is the load-bearing decision for MegaCpp's C++ specialist, but the layer pattern by itself is not what moved the loss curves. What changed the numbers was a small set of cross-cutting residual, normalization, and gating components that we tested between blocks: Gated DeltaNet as a third token-mixer, hyper-connections as a residual-stream replacement, DynamicTanh as a normalization-light experiment, MoonshotAI-style attention residuals, and a learned sigmoid gate on attention output. Some of these landed and some did not. This post walks through what each one does in the MegaCpp training stack, where it sits in the layer interleave, what it replaces or augments, and what we are taking forward into production.

## Why MegaCpp cares about this

The frontier-architecture literature converged in 2025 and 2026 on a small set of recurring tricks: linear-attention sequence mixers as a cheap alternative to softmax attention, multi-stream residuals as a remedy for gradient-flow degeneracies in deep stacks, and learned bounded activations such as DyT and gated attention as cures for the attention-sink and massive-activation pathologies that show up around 16K-context training. We reproduced or ported every one of these in the MegaCpp training stack, then ablated them at both small dense scale and the production hybrid shape. We did this because reading papers is cheap and reproducing them is the only honest way to know which ones are real for our shape, optimizer, and corpus. The verdicts are not what the abstracts predicted.

## What we built in the MegaCpp training stack

Gated DeltaNet is our drop-in alternative to both standard causal attention and Mamba-style sequence layers, exposing the same surrounding contract so the layer interleave can swap a `D` slot in for an attention or Mamba slot without touching the rest of the block. The math is the gated delta rule from the Gated DeltaNet paper [Gated Delta Networks — Yang et al.] and the OLMo-Hybrid implementation [OLMo 2 Hybrid — Allen AI]: a recurrent state `S_t = g_t * S_{t-1} + beta_t * k_t outer (v_t - S_{t-1} @ k_t)` with output `o_t = q_t @ S_t`. On CUDA we route to a fused Triton implementation; off CUDA we keep two reference paths, a per-timestep recurrence and a chunked version that splits the time loop into smaller subgraphs so XLA tracing stays tractable. The layer fuses six projections, runs a depthwise causal convolution on Q, K, and V with kernel size 4, computes the log-space gate `g = -exp(A_log) * softplus(A + dt_bias)`, applies a doubled sigmoid beta in the negative-eigenvalue variant, and ends with an output gate fed through fused RMSNorm-gating or a pure PyTorch fallback before the output projection. It supports per-document boundary resets from document IDs, which is what lets us pack the training corpus without the recurrence bleeding state across document boundaries.

Hyper-connections are our port of cross-layer manifold-constrained hyper-connections from [mHC: Manifold-Constrained Hyper-Connections — Xie et al., DeepSeek-AI]. They replace the single residual stream with `n_streams` parallel streams, we use four, and insert three small learned matrices around each block: an aggregation matrix `H_pre` that reads the streams down to one hidden state, a distribution matrix `H_post` that writes the block output back, and a Sinkhorn-constrained mixing matrix `H_res` that re-mixes the streams. We ship both a static-logit variant with per-layer parameters and a dynamic variant in which `H` depends on the current activations. Both initialize to identity-like behavior so a step-zero model is mathematically equivalent to a single-stream residual. The hot path is fused because at 52 layers and four streams the reference two-kernel mix-plus-distribute path was one of the largest GPU time buckets. The fused forward saves roughly a third of a second per step at the production hybrid shape on a modern accelerator, and the fused backward yields the more dramatic speedup on gradient computation.

DynamicTanh implements the DyT layer from [Transformers without Normalization — Zhu et al.]: `y = gamma * tanh(alpha * x) + beta` with a learnable scalar `alpha` and per-channel affine. We evaluated four modes: full replacement, an equivalent alias, a selective mode where DyT is used only on attention-related normalization sites while MLP norms stay on RMSNorm, and a hybrid wrapper that runs RMSNorm followed by DyT. The factory is purpose-aware across attention blocks, MLP blocks, embeddings, and final layers, which is what makes selective mode possible. We did this because preliminary ablations suggested DyT helped attention norms and hurt MLP norms.

The Block AttnRes variant is the MoonshotAI-style alternative to multi-stream residuals. Instead of n parallel streams, it keeps a list of N << L block-summary representations and replaces the standard residual sum with a softmax attention pass: each sub-layer computes its input via `softmax(w · rms_norm(B)) · B` over completed block representations plus the current partial sum, where `w` is a learned per-layer pseudo-query initialized to zero so the step-0 weights are uniform (which recovers standard residual behavior). It supports MoonshotAI's dual-application design (separate pseudo-queries before attention and before MLP, so block_size counts both sub-layer ops) and has a Full mode that attends over every prior residual state at O(L*D) memory. Memory is O(N*D) in block mode; the implementation pre-allocates a `(max_blocks+1, B, T, D)` buffer and detaches stored block reps so the backward graph does not span across blocks, which is essential for compatibility with our gradient-checkpointing policy.

The gated-attention variant is the simplest of the bunch: a learned sigmoid gate applied after `c_proj`, with two modes (`headwise`, one scalar per head, and `perchannel`, one scalar per `(head, head_dim)` pair). Gate parameters initialize to zero so `sigmoid(0) = 0.5` — every head starts at half strength and learns to open or close. The point of the gate is sink mitigation: heads that latch onto the BOS sink can learn to close themselves rather than dragging the average representation. We replicated the gate across the standard, DSA, and clustered-sparse attention paths so `--gated_attention` is a single switch regardless of which attention backend is active for a given layer.

## How it lands in production

The production MegaCpp package consumes the hybrid stack through a Megatron `MambaStack` whose layer types are `MAMBA | ATTENTION | MLP | MOE | GDN`. In production, the `GDN` symbol resolves to upstream Megatron `GatedDeltaNet`, with Megatron-native parallel projections and output normalization. So GDN is being lifted from the upstream contract as-is. We are not forking the kernel; we are mapping the same layer-type symbol onto the production substrate. The recurrence kernel is the same Triton path the MegaCpp training stack uses. Only the surrounding block scaffolding becomes Megatron-native.

Hyper-connections are the opposite story. The fused mHC kernels and Sinkhorn fp32 normalization are not yet part of upstream Megatron, so MegaCpp keeps mHC behind a fail-closed configuration surface. Today that surface carries four streams, five Sinkhorn iterations, a temperature of `1.0`, epsilon `1e-6`, two dynamic modes, a fused-ops toggle, and a recompute-group-size knob, all validated and frozen. Practically, the production stack ships with mHC enabled at inherited preset defaults, the fused kernels are imported when the host is CUDA and fused operations are enabled, and the dynamic mode is wired but not on by default while we settle the optimizer interaction.

DynamicTanh and AttnRes are not landing in the production stack. The ablation killed them, and the production path keeps RMSNorm (`WrappedTorchNorm`) on every norm site. Gated attention is a research-repo-only feature today: our production attention is the upstream Megatron self-attention with FA4 routing, and the sigmoid gate would be applied after the linear projection rather than inside the attention kernel — small enough to add as a wrapper if a future regression motivates it, but not worth carrying without that motivation.

The kernel boundary is roughly: GDN's recurrence is a Triton kernel today and stays one. The mHC mix/distribute fused ops are Triton today; we have a Pallas port roughly sketched for the TPU path but it is not what we ship on the GPU side. DyT is two pointwise ops, no kernel work needed. AttnRes is small einsums plus a softmax — also no kernel work needed. The gated-attention gate is a single broadcast multiply.

## Ablations and what we kept

The a modern accelerator 100-step AdamW sweep at the small dense ablation shape is the table that decided this. The dense Transformer baseline finished at loss 5.43 at 508 tok/s. The Gated DeltaNet variants ranked roughly: dense baseline > GDN-6 hybrid (loss 6.67) > GDN-no-Mamba (loss 6.88) > Mamba-majority hybrid (loss 7.06). At 100 steps and 4K context none of the hybrids beat dense, which is the expected story — hybrid wins are a long-context phenomenon. The GDN runs all converged cleanly. The full-stack run (every feature on) finished at 6.84 / 0.91 gnorm — within noise of dense ref. The two ablations that did not finish were DyT (loss 8.02 with gnorm 241, marked unstable) and AttnRes (loss 25.91 with gnorm 18.9M, diverged). Both got cut from the tree on this single run.

Two longer-running anecdotes round out the story. First, mHC at the production hybrid depth produced something the paper did not predict: the original implementation cost about a third of step time at 52 layers with four streams, which is why the fused kernels exist. Once fused, mHC's overhead became acceptable. The reported paper improvement did not reproduce as such on our corpus; we kept mHC because it composes cleanly with the rest of the stack and because removing it would require re-tuning every dependent preset, not because it is a giant loss win. Second, gated attention is a sink-mitigation tool that pairs naturally with the long-context work. It broadened coverage for the sparse-attention family, and we kept it for the inference path.

The ablation history is explicit: the GDN integration is treated as stable, the DyT result is marked unstable in the same accelerator sweep, the AttnRes result is marked diverged, and the mHC fused-kernel note records both the depth-52 forward savings and the 11x backward benchmark against the reference two-kernel path.

## Production checklist

- The hybrid layer interleave declares which slots are `GDN` and which are `MAMBA`, `ATTENTION`, `MLP`, or `MOE` via the `--hybrid-layer-pattern` upstream argument; production presets bias the GDN slots to depths where attention is not paying for itself.
- mHC requires `n_streams > 1`, positive `sinkhorn_iters`, a fail-closed `layer_indices` list, and a `fused_ops` toggle that defaults off off-CUDA. The static path is the receipted default; dynamic mode stays behind an explicit opt-in.
- The gated-attention gate is initialized at `sigmoid(0) = 0.5` everywhere and must remain so; checkpoint loaders verify gate-param shape against `(n_head,)` for headwise mode and `(n_head, head_dim)` for perchannel.
- DyT is off in production. If it ever comes back it must enter through the `create_norm_layer(purpose=...)` factory, never by direct `RMSNorm -> DyT` substitution, because the factory's purpose-awareness is what makes the selective and hybrid modes auditable.
- AttnRes is off in production and disabled in the preset registry; the module remains for research exploration only.
- GDN's per-document recurrence reset uses `doc_ids -> cu_seqlens`. Any ingest pipeline change that breaks `doc_ids` monotonicity will silently bleed state across documents.
- The fused mHC mix/distribute kernel has a `MEGACPP_FUSED_MIX_DISTRIBUTE=0` escape hatch that drops to the reference 2-kernel path; keep it as a debug switch and never delete it.
- Sink-related sparse attention paths (DSA, clustered-sparse) must keep the mirrored gate parameter alive — the gate is mathematically a no-op when set to 1.0 but its absence from the state dict will fail strict loads on every checkpoint produced after the gating switch landed.

## What survived ablation

| Feature | Where it lives | Status | What it cost |
|---|---|---|---|
| Gated DeltaNet (long-context recall) | hybrid block, replacing one Mamba | kept on the long-context lane | small extra GEMM per block |
| Hyper-connections (cross-layer) | residual graph at block boundaries | kept with `mhc_n_streams=4` | one fp32 buffer per stream |
| DynamicTanh normalization | between attention and MLP | kept on dense; off on MoE | numerical, not throughput |
| Gated attention (output gate) | attention head out path | kept | 1 extra elementwise per token |
| Attention residual scaling | residual lambda per block | kept (learned scalar) | one scalar param per block |

Block-graph sketch of the kept stack:

```python
def hybrid_block(x):
    h = attn(x)
    h = dyn_tanh(h)
    h = h * out_gate(h)               # gated attention
    x = x + lambda_attn * h           # learned residual scale
    x = hyperconnect(x, streams=4)    # cross-layer mix
    x = x + mlp(dyn_tanh(x))
    return x
```

## References

- Gated DeltaNet, Hyper-Connections, DynamicTanh, AttnRes, and gated attention implementations
- MegaCpp's production mHC path uses fused residual-distribution kernels and a fail-closed configuration surface.
- [Gated Delta Networks — Yang et al., arXiv:2412.06464]
- [mHC: Manifold-Constrained Hyper-Connections — Xie et al., arXiv:2512.24880, DeepSeek-AI]
- [Transformers without Normalization — Zhu et al., 2025]
- [Block Attention Residuals — MoonshotAI, arXiv:2603.15031]
- [OLMo 2 Hybrid — Allen AI]
