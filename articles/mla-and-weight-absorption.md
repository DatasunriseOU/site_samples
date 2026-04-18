---
title: "MLA Weight Absorption: What We Kept, What We Dropped for the C++ Specialists"
description: "Multi-Head Latent Attention in production: why DeepSeek's absorbed decode path is the right choice for KV-cache, why it is the wrong choice for training, and how the C++ specialist ensemble uses both."
date: 2026-04-18
author: "Boris Tamarkin"
tags: ["mla", "attention", "deepseek", "flash-attention", "kv-cache", "training", "inference"]
---

# MLA Weight Absorption: What We Kept, What We Dropped for the C++ Specialists

Multi-Head Latent Attention is the one piece of the DeepSeek-V3 architecture that keeps showing up in every candidate attention stack for MegaCpp's specialist SLMs. The draw is obvious: a compressed latent `c_kv` an order of magnitude smaller than concatenated K/V, a KV-cache that fits in one B200 for sequences that would otherwise need two, and a numerically clean separation of the positional (RoPE) and content (NoPE) parts of the query. The trap is that the public DeepSeek inference path does not just use MLA, it uses a specific reformulation called **weight absorption**, and that reformulation is correct for decode and wrong for training. This post walks through the analysis we did to pin that down, and which pieces of MLA survived into the C++ specialists.

## What MLA looks like when we train it

The straightforward MLA path is what our POC training code does today. The KV projection produces a low-rank latent:

```
c_kv   = W_dkv @ x              # (B, T, kv_lora_rank)
K_nope = W_uk  @ norm(c_kv)     # (B, T, H, d_nope)
V      = W_uv  @ norm(c_kv)     # (B, T, H, d_v)
K      = concat(K_nope, K_rope)
```

The full `K` and `V` are materialised per layer, Flash Attention consumes them as three dense tensors with the standard shapes, and `softmax(Q K^T / sqrt(d)) V` is fused into one kernel with O(T) activation memory. With `recompute_kv_upproj=True` (our training default), only `c_kv` (`[B, T, kv_lora_rank]`) and `K_rope` (`[B, T, 1, d_rope]`) are saved for backward; `K` and `V` are regenerated from the latent in the backward pass. This is a classic activation/FLOP trade and it works cleanly with every dense backend: FA3, FA4, FlexAttention, Pallas/Splash, SDPA, and the manual fallback. The numerical model is one attention operator per block with one set of Q/K/V inputs and one softmax.

## What weight absorption is, algebraically

DeepSeek's inference code carries a second path, `attn_impl != "naive"`, that rewrites the score computation by absorbing the KV up-projection into Q and keeping attention in latent space. The identity is basic linear algebra:

```
Q_nope @ K_nope^T  =  Q_nope @ (W_uk @ c_kv)^T
                   =  (Q_nope @ W_uk^T) @ c_kv^T
```

Let `Q' = Q_nope @ W_uk^T`. Attention over the NoPE part is now `Q' @ c_kv^T` instead of `Q_nope @ K_nope^T`. The value side absorbs the same way:

```
output  =  attn_weights @ V
        =  attn_weights @ (W_uv @ c_kv)
        =  (attn_weights @ c_kv) @ W_uv
```

The net result is that the KV-cache stores `c_kv` (`kv_lora_rank` dims) plus `K_rope` (`d_rope` dims per KV head), and never materialises the full per-head K and V tensors. For decode of one token against a long history that is exactly the right optimisation: a DeepSeek-V3-scale cache drops from the multi-megabyte per-token regime to the sub-megabyte regime, and the sequence-length axis is walked over a compact latent instead of a fat K/V.

The price is that the attention score is no longer a single `Q @ K^T`. It is a sum of two terms that must be combined before softmax:

```
scores = (Q' @ c_kv^T) + (Q_rope @ K_rope^T)
```

Softmax is not distributive over addition, so the two terms have to be summed inside whatever kernel computes attention. No public Flash Attention variant supports this split-score contract. FA3 and FA4 expect a single Q/K/V triple. FlexAttention exposes a `score_mod` hook but not a "pre-softmax additive score from a second QK product". Pallas/Splash has the same single-product assumption. SDPA has the same assumption. Therefore, to use the absorbed form in a fused kernel you would have to write that kernel yourself.

## The FLOP argument, for training

With DeepSeek-V3 defaults (H=128 heads, `d_nope`=128, `d_rope`=64, `d_v`=128, `kv_lora_rank`=512) the attention core FLOPs separate cleanly. The standard expand-then-attend path spends `2*B*H*T^2*320` FLOPs in the attention core: `qk_head_dim=192` on the `Q @ K^T` product plus `v_head_dim=128` on `attn @ V`. The absorbed path pays attention against the full `kv_lora_rank` on both sides: `2*B*H*T^2*512` for the NoPE score, `2*B*H*T^2*64` for the RoPE score, and `2*B*H*T^2*512` on the value side, for a total of `2*B*H*T^2*1088`. The projection FLOPs (`W_uk` absorption into Q plus the `W_uv` projection after attention) equal the FLOPs saved by skipping the explicit KV up-projection, so projections net to zero.

The ratio is `1088 / 320 = 3.4x` more attention core FLOPs under absorption. For our NAM heads (H=24, `kv_lora_rank`=512) the constant changes but not the ratio; the attention core is still 3.4x heavier under absorption. The ratio improves when `kv_lora_rank` drops, but `kv_lora_rank` must stay at least as large as `d_nope` (and in practice 4x larger) for the low-rank approximation to carry the representational weight it is supposed to carry.

The activation-memory argument is worse. In standard MLA with `recompute_kv_upproj=True`, the saved tensors are `c_kv` and `K_rope`, a few tens of megabytes. Under absorption we would still save those, plus the expanded `Q' = Q_nope @ W_uk^T` at shape `(B, T, H, kv_lora_rank)`, which is larger than standard `Q` at `(B, T, H, qk_head_dim)`. More importantly, without Flash Attention we would have to save the materialised `(B, H, T, T)` attention-weight tensor to backprop through the split-score softmax. At B=8, H=24, T=8192 in BF16 that alone is 25 GiB per block. Catastrophic is the technical term.

The training verdict is unambiguous: weight absorption is a decode optimisation. For training on the kind of sequence lengths our specialists see (4K up to 64K packed context graphs from the v4 context-graph sampler), expand-then-attend with Flash Attention and latent-only activation save is strictly cheaper on both FLOPs and memory.

## Why inference is the opposite story

The inference regime inverts every term in that calculation. Decode runs at `T_query = 1` against a `T_kv` that grows to tens of thousands of tokens. The attention core FLOPs scale as `T_query * T_kv`, so the 3.4x constant sits on top of a tiny number. The activation-memory argument vanishes entirely because decode does not backpropagate; there is no attention-weight tensor saved for backward, and there is no Flash Attention benefit to forfeit because the incremental decode kernel is a trivial softmax over the history. The only thing that actually scales is the KV-cache itself, and that is exactly what absorption shrinks.

Concretely, under absorption the per-token KV-cache entry is `c_kv` plus `K_rope`: `kv_lora_rank + H_kv * d_rope` scalars. Under the standard path the cache entry is per-head `K` and `V`: `H * (d_nope + d_rope) + H * d_v` scalars. For DeepSeek-V3-scale attention heads, absorption buys roughly an order of magnitude. For a 1M-token C++ context that is the difference between a KV-cache that fits in one B200 and a KV-cache that does not fit in two.

A secondary benefit for the serving path is that `c_kv` is a single low-rank tensor, which composes cleanly with paged KV-cache, with block-sparse attention, and with the attention-sink mitigations described in the long-context post. Paged allocation on a 512-dim latent is a lot easier than paged allocation on a 384-dim per-head `K` plus a 128-dim per-head `V` across H heads.

## What survived into the C++ specialists

MegaCpp's specialist ensemble uses MLA in two forms, chosen per regime.

| Regime        | Form              | KV-cache entry               | Kernel path         |
|---------------|-------------------|------------------------------|---------------------|
| Training      | expand-and-attend | n/a (no cache)               | Flash Attention     |
| Serving       | absorbed          | `c_kv` + `K_rope` per token  | split-score softmax |
| Long-ctx eval | absorbed          | `c_kv` + `K_rope` per token  | split-score softmax |

**Training uses standard MLA.** Every specialist is trained with the expand-then-attend path, `recompute_kv_upproj=True`, and Flash Attention on the dense attention blocks. The KV projection generates `c_kv`; `K_nope`, `V`, and the concatenated `K` are produced on the fly; only `c_kv` and `K_rope` are saved for backward. On the MoE-heavy production hybrid this is paired with the MLA-up-proj selective recompute (`--recompute-modules mla_up_proj`), which is one of the named modules in the golden configuration. The reason this works is that MLA up-projection is cheap to recompute from the latent and expensive to store in activations; it is the exact kind of operation selective recompute was built for.

**Inference uses absorbed MLA.** For the serving path and long-context eval we run the absorbed form: Q absorbs `W_uk`, the KV-cache stores only `c_kv` and `K_rope`, attention is computed in latent space as a sum of the NoPE and RoPE score terms, and the `W_uv` projection happens after the attention-weight has been multiplied with `c_kv`. The split-score softmax is handled in a custom path; the usual Flash Attention kernels are not involved, which is fine because decode is not the phase whose wall-clock we are protecting.

We explicitly do not mix the two. The checkpoint format is the standard one: `W_dkv`, `W_uk`, `W_uv`, and the RoPE projections as distinct tensors. The inference loader reshapes `W_uk` into the per-head `[H, d_nope, kv_lora_rank]` layout and fuses it into the Q path once at model load; nothing about the trained weights changes. This has the useful side-effect that a single checkpoint can drive either path, and we can A/B the absorbed inference path against the naive inference path without retraining.

## The parts that did not survive

A few variants that sounded plausible did not make it into the specialist stack.

We considered absorbing at training time for "consistency" with the inference path. Rejected on the 3.4x attention-core FLOP penalty, the activation-memory blow-up, and the Flash Attention incompatibility. There is no version of the cost curve where absorbed training is cheaper than expand-then-attend training on our sequence lengths.

We considered a custom Flash-Attention-shaped kernel that fuses the split-score softmax. This is tractable on paper (the two score products can share the same tiling schedule if the RoPE term is cached per KV block) but the engineering budget against the FLOP win is terrible: attention is a single-digit percentage of the production hybrid's compute breakdown, and an absorbed kernel that only covers the forward is useless without a matching backward, which no public kernel (FA3/FA4, CUTLASS hopper_fmha, TileLang `sparse_mla_bwd`) supports. FA3/FA4 backward is documented as "no plans, accuracy open problem" by upstream. For the specialists, writing a proprietary FP8-plus-split-score attention backward is not where the hours go.

We considered a quantised `c_kv` cache (FP8 e4m3fn) for decode. This is on the roadmap but not in production. The per-token scale granularity interacts with the RoPE term in a way we have not finished measuring, and the current FP8 attention backward literature is empty; the inference-only variant is doable but has to carry its own calibration pass.

## What the analysis leaves us with

The MLA absorption question is the clearest example in our stack of a transformation that is simultaneously an optimisation and a de-optimisation depending on which axis you measure. Train with the standard path because FLOPs and Flash Attention memory dominate. Serve with the absorbed path because KV-cache dominates. Keep one checkpoint format and switch regimes at load time, not at training time. And never read a single-kernel microbenchmark as a claim about full-model throughput; we learned that lesson the expensive way on the FP8 Mamba scan, and this was the cheap version of the same mistake waiting to happen.

For the C++ specialists specifically, the combination of MLA training + absorbed MLA decode + the v4 context-graph sampler is what makes 64K repo-level reasoning affordable at inference. The decode KV-cache is small enough that one specialist can hold the full context of a realistic translation unit plus its headers and call graph on a single GPU; the training path is fast enough that we can retrain any of the eight specialists on a week of calendar time per checkpoint. Neither property holds under the naive combination of the two.

## References

- `24-mla-weight-absorption-analysis.md`
- `deepseek_mla_strategy.md`
- `06-long-context.md`
- internal training review notes
- `architecture_and_eval_en.md`
- `v4_architecture.md`
