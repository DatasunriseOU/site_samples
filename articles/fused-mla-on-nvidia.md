---
title: "Fused MLA on Hopper and Blackwell: projection, RoPE, and the KV cache that ships"
description: "The NVIDIA side of Multi-Latent Attention in the MegaCpp ensemble: a fused down-norm-up projection, a fused split-RoPE-concat Triton kernel, a compressed KV cache, and how it all lands on Megatron-Core."
date: "2026-04-18"
tags: ["mla", "triton", "h200", "blackwell", "rope", "kv-cache", "fused-kernels"]
---

The cross-path MLA story (which parts of DeepSeek-V3 we trained with, why weight absorption is the wrong choice for the training path, what survived into inference) is already covered in our [MLA Weight Absorption](/blog/mla-and-weight-absorption) post. This one is narrower: how MLA runs on NVIDIA once you care about kernels. Fused down-projection / RMSNorm / up-projection. Fused split / RoPE / concat in one Triton kernel. A compressed KV cache that stores `c_kv` plus a single RoPE'd key row instead of full `K`/`V`. And the Blackwell tensor-core path through TransformerEngine. Everything here is the part of MLA that only matters if you are pointing it at H200 or GB10.

## Why MegaCpp cares about this

At specialist scale the attention block is still the compute cost we optimise hardest, and MLA's compressed latent makes it a numerically clean place to fuse things. The training path is unambiguous: expand, Flash Attention, done. Where kernels start to matter is between the projection, the norm, the split into nope / rope, the rotary application, and the recombine. On H200 that sequence without fusion is five kernel launches per attention per layer per microbatch step; on a depth-56 hybrid preset that becomes the kind of cost we do not want to pay twice. The other question - whether MLA pays off for the *specialist* SLM sizes at all - is what the compressed KV cache answers: 4x smaller KV cache at 4-bit quantization relative to bf16 is a deployment unlock for our on-device inference targets.

## What we built in the public MegaCpp MLA path

`mla.py` is the reference training path. It takes `x`, does the down-projection (either separate `w_dq` + `w_dkv` or a fused `w_dqkv`), RMSNorm on the latent, up-projection through `w_uq` and `w_ukv`, reshape to `(B, T, H, qk_nope_head_dim + qk_rope_head_dim)` on the Q side, split the KV side into `low_rank_main` (`kv_lora_rank`) and `low_rank_rope` (`qk_rope_head_dim`), apply partial RoPE only to the rope portion, concatenate back, and hand the lot to Flash Attention. `recompute_kv_upproj=True` is the default training knob: backward recomputes the large `H * (d_nope + d_v)` tensor from the small `kv_lora_rank` latent. `recompute_q_upproj` is optional and wraps norm / up-project / split / RoPE / concat in one `cp.checkpoint` so inductor has the right adjacency to fuse RMSNorm and matmul into a single Triton kernel on the compile path. We explicitly do not use the weight-absorbed training path; the rationale is spelled out in the dedicated post, and the one-line summary is that the FLOP increase and the loss of Flash Attention compatibility dominate the KV-memory win that only applies at decode.

The fused projection is `fused_mla_projection.py`. `FusedDownNormUp` is a standalone `autograd.Function` that fuses `y = W_ukv @ RMSNorm(W_dkv @ x)`, saving `(x, rrms, w_dkv, w_ukv, rms_weight)` for backward and recomputing `latent = W_dkv @ x` and `normed = latent * rrms` during the backward pass. The point is memory: the large `kv_lora_rank`-sized latent is not held across the block, only the tiny per-token `rrms` scalar is. Backward reassembles the full RMSNorm gradient formula directly (`grad_latent = rrms * (grad_normed - latent * rrms^2 * dot / d)`) so we do not pay for an `autograd.grad` pass through `F.rms_norm`. An important caveat: this is a standalone building block and an experimental unit-test target. The live `MultiLatentAttention.forward` path in `mla.py` still uses the separate down / norm / up chain, and the `mla_fused_down_proj` flag in the config enables only a single-matmul fused *down*-projection (`w_dqkv`), not the full down+norm+up fusion. Megatron-Core has the same seam on a different axis with its `LayerNormColumnParallelLinear`; on our path today inductor fuses RMSNorm + matmul into a single Triton kernel (`triton_tem_fused__fused_rms_norm_mm_t`) when we get the adjacency right in the checkpoint wrapper, which is the reason `_q_norm_upproj_and_rope` is laid out the way it is.

Fused RoPE lives in two Triton kernel surfaces: the MLA-specific path and the generic dense-attention path. The MLA version is the hotter one. The reason MLA needs its own fused RoPE kernel is the partial-RoPE layout: only `rope_dim = qk_rope_head_dim // 2 * 2` of each head gets rotated, the rest pass through. A naive PyTorch path allocates three intermediate tensors per application (`q_nope`, `q_rope`, rotated `q_rope`) and `torch.cat`s them back. The fused Triton kernels in `fused_mla_rope.py` - Q-side forward and backward, KV-side forward and backward, plus the packed `thd` layout variant - do the split, the rotation, and the concat in a single grid launch. Autotune covers `BLOCK_H` in `{1, 2, 4, 8, 16, 32, 64, 128}` keyed by `emb_dim` and `head_num`. The kernel operates in place on Q for forward and on `DO` (grad output) for backward, and the backward formula is the transpose of the forward rotation: `dx1 = dy1*cos - dy2*sin`, `dx2 = dy1*sin + dy2*cos`. Packed `thd` mode uses `cu_seqlens` to walk the sequence boundaries per token and find the right cos/sin row. On the bthd path the token index is just `pid_m % seq_len`.

The generic (non-MLA) fused RoPE for Q and K lives in a separate Triton kernel from the MLA-specific one because the layout is different: it rotates the full head dimension, not a rope subportion, and it groups `ROPE_GROUP_SIZE=4` heads per thread block so cos/sin loads are shared. With GQA it specializes on `n_heads_q` and `n_heads_k` as `tl.constexpr` and skips the K write when the head index exceeds `n_heads_k`. The split-half convention is the same as the MLA kernel (`y1 = x1*cos + x2*sin`, `y2 = -x1*sin + x2*cos`), which means we maintain one RoPE numerical contract across both kernels and the reference partial-rotary implementation. The long-context piecewise and YaRN RoPE variants are precomputation-only and feed both the fused and reference paths.

Weight absorption specific to the fused NVIDIA path is subtle. The classic DeepSeek inference path absorbs `W_uk` into `Q` and `W_uv` into the output projection, leaving attention to operate in `kv_lora_rank` dims. On NVIDIA the interesting thing is that Megatron now has an absorbed-weights MLA variant, and the public MLA shared-runtime sample has explicit adapter hooks for it. We do not route training through the absorbed variant; we do allow it for inference spec selection when the decode shape is the dominant workload, because it changes the KV cache from storing `K` and `V` expansions to storing the `kv_lora_rank` latent plus the single rope'd key row. The key algebraic identity is simple (`(Q_nope @ W_uk) @ c_kv^T == Q_nope @ (W_uk @ c_kv)^T`), but the kernel consequence is that FA3/FA4 cannot fuse the absorbed path, and the attention dot products move into `kv_lora_rank=512` dims rather than `qk_head_dim=192`, which is why it is strictly a decode-side choice.

The KV cache layout for training-shaped decode is `MLAKVCache` in `mla.py`. Per layer, two buffers: `low_rank_main` of shape `(B, T_max, kv_lora_rank)` and `rope_caches` of shape `(B, T_max, 1, qk_rope_head_dim)`. The RoPE key is cached once per token, broadcast across the H query heads at read time. `update` enforces uniform `cache_seqlens` across batch (because T_new is a single position advance per update call), writes at `self.cache_seqlens[0]`, and `advance` bumps the counter after *all* layers have updated. That last invariant - advance the counter once after the full layer stack, not per layer - is the thing that tripped the earliest bring-up, because per-layer advance would have silently written into the same cache slot N times.

The quantized variant downstream of the compressed cache uses PolarQuant and TurboQuant. Both reach roughly 3.8x compression at 4 bits versus bf16 with minimal quality loss; TurboQuant (random orthogonal rotation plus per-coordinate scalar quantization) is the successor we prefer, and PolarQuant is retained for analytical guarantees we still rely on for one specialist.

## How it lands in MegaCpp

the public MLA shared-runtime sample is an adapter surface. One adapter wraps the standard `MLASelfAttention`, one wraps `FusedMLASelfAttention` (the Transformer Engine tensor-core path), and one wraps the upstream absorbed-weights variant when it is available. The adapters exist for one reason: to pass a pipeline-layer offset to the underlying class when the upstream constructor supports it, so pipeline-parallel stage placement lines up without forking the MLA implementation. The adaptation step rewrites the GPT layer spec in place instead of rebuilding the whole object graph.

The attention layer spec is built by asking the current transformer implementation for the right attention submodules and then threading the MLA-relevant options through unchanged. On the Transformer Engine path that includes QK layernorm, multi-latent attention, QK L2 norm, TE op fusion, Kitchen integration, TE activation selection, and the MLA down-projection fusion flag. The point is that every knob that matters for MLA fusion is a passthrough, not a MegaCpp reinvention; we own the adaptation seams, Megatron owns the module internals.

The hybrid MLA / DSA interleave is a MegaCpp-specific layout. The deep-hybrid full spec selects MLA for some attention ranks and DSA for others; MTP (multi-token-predict) layers always get MLA, never DSA, because the MTP head needs dense attention semantics the sparse path does not provide. Each branch builds its own attention spec through the same shared builder logic.

Blackwell tensor-core path: H200 and GB10 both route MLA through TransformerEngine. H200 gets the full `FusedMLASelfAttention` bf16/fp8 path; GB10 pins `disable_rht=True` because the Random Hadamard Transform is not stable on that target, which shifts MLA QK numerics enough that GB10 results are tracked separately from H200 results.

What lands as-is: the MLA module, the fused MLA RoPE Triton kernels, the MLA KV-cache path, and the PolarQuant and TurboQuant wrappers. What gets lifted but guarded: the standalone fused down-norm-up path, still experimental. What moves to Megatron: the attention layer spec, the distributed-optimizer integration, TP/PP/SP wiring, and the FP8 communication layer. The weight-absorbed training path is intentionally dropped.

## Design choices that survived validation

The fused MLA RoPE kernels landed as part of a larger Megatron-optimization wave (alongside fused Mamba conv, PP with parameter-count-weighted stage partitioning, TP all-reduce overlap, sequence parallelism for norm/dropout, ZeRO-1 distributed optimizer, FP8 comm, and EP load balancing). MLA-specific bugs we fixed on the way: latent-dim mismatch when resuming from non-MLA checkpoints, TP all-reduce placement for MLA's asymmetric Q/KV projections, FlexAttention `score_mod` ignoring per-head RoPE frequencies, and gradient checkpointing interacting with the in-place fused MLA RoPE kernel. We also tried weight-absorbed training: it lost on FLOPs (attention dot products move from `qk_head_dim=192` to `kv_lora_rank=512`) and on Flash Attention compatibility (absorbed attention has to compute and sum nope/rope score components separately, which breaks every fused FA kernel we use). Autotune for `fused_mla_rope` lands at `BLOCK_H=16` or `32` on H200 depending on `emb_dim`; on GB10 the kernel is bandwidth-bound and autotune prefers a larger block to amortise loads.



We did try the weight-absorbed path for training. It lost on two counts: FLOP increase (~2.7x for the attention O(T^2) dot products because they operate in `kv_lora_rank=512` dims instead of `qk_head_dim=192`) and Flash Attention incompatibility (absorbed attention must split the nope and rope score components, compute them separately, and sum before softmax, which breaks every fused FA kernel we care about: FA3, FA4, FlexAttention, Pallas, Splash). The memory saving that justifies absorption at decode is immaterial during training because `recompute_kv_upproj=True` already means the large K/V tensors are never stored.


## Production checklist

- Training path is expand-then-attend; absorbed MLA is an inference-side choice only, gated through the Megatron adapter in `mla_shared.py`.
- `recompute_kv_upproj=True` is the default whenever Flash Attention is the backend.
- `fused_mla_rope=True` is the NVIDIA default; the kernel mutates Q in place, so checkpoint placement must preserve the pre-rotation tensor.
- `MLAKVCache` is the only supported KV cache for MLA, and `MLAKVCache.advance` is called once per step after all layers have updated.
- `FusedDownNormUp` is experimental; the live path runs the separate down / norm / up chain and relies on inductor to fuse `rms_norm + mm` via adjacency.
- On GB10 the recipe pins `disable_rht=True` and MFU is computed against NVFP4 peak.
- KV quantisation (PolarQuant / TurboQuant) is inference-only; `attn_layer_map` routes hybrid stacks through the right quantiser per layer type.

## What MLA fuses on NVIDIA

| Stage | Naive launches per attention | Fused path | Memory note |
|---|---|---|---|
| Down + RMSNorm + Up | 3 GEMMs + 1 norm | `FusedDownNormUp` autograd.Function | only `rrms` saved, latent recomputed in bwd |
| Q-side split + RoPE + concat | 3 ops + cat | MLA-specific Triton kernel | in-place on Q, autotune over `BLOCK_H` |
| KV-side rope row + concat | 2 ops + cat | KV-side fused kernel | one RoPE'd key row stored, not full K |
| KV cache (decode) | full `K`, `V` | compressed `c_kv` + 1 RoPE row | ~4x smaller at 4-bit |

The fused projection's autograd contract:

```python
class FusedDownNormUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_dkv, rms_weight, w_ukv, eps):
        latent = x @ w_dkv.T
        rrms = torch.rsqrt(latent.pow(2).mean(-1, keepdim=True) + eps)
        normed = latent * rrms * rms_weight
        y = normed @ w_ukv.T
        ctx.save_for_backward(x, rrms, w_dkv, w_ukv, rms_weight)
        return y
```

## References

- [Megatron-LM — NVIDIA, GitHub](https://github.com/NVIDIA/Megatron-LM)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [PolarQuant](https://arxiv.org/abs/2502.02617)
- [TurboQuant](https://arxiv.org/abs/2504.19874)
- [Hybrid layout notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md)
