---
title: "MoD, MoDA, and MTP: Dynamic Depth and Multi-Token Heads"
description: "How we allocate compute per layer with Mixture-of-Depths, cross-attend across layers with MoDA, and train multi-token prediction heads that double as a draft source for self-speculative decoding."
date: "2026-04-18"
tags: ["mixture-of-depths", "moda", "mtp", "speculative-decoding", "training"]
---

Three features in the hybrid stack all sit at the "what does each layer do for each token" layer of the design: Mixture-of-Depths (MoD) decides which tokens a layer actually processes, MoDA lets attention heads reach across layer depth, and Multi-Token Prediction (MTP) trains a shared block to predict several future tokens per step. They overlap enough that people conflate them; they are genuinely different. This post separates them, says what we shipped per specialist, and explains how MTP weights do double duty as a draft source for speculative decoding (the acceptance/verifier side is covered in a separate post on ensemble speculative decoding, and this one does not repeat that content).

## Why MegaCpp cares about this

Training compute is not uniform across tokens. A whitespace token, an opening brace, a common identifier like `i` — none of them deserve the full residual path through every layer. A function-name token in a long definition, an `if` condition, a template parameter — those do. MoD formalises that intuition: put a lightweight router in front of each block and let it decide which tokens take the full path versus a residual skip. The original formulation (Raposo et al. 2024, "Mixture-of-Depths") used top-k selection; we explored three routing modes and kept the ones that trained stably.

MoDA is a different axis. Instead of "do I run this block for this token", it's "can my attention heads reach keys/values from earlier layers at the same position". That gives the stack cross-depth context at negligible parameter cost, because most layers already materialise K/V for attention anyway.

MTP is the third. At train time it is an auxiliary loss head that predicts multiple future tokens through a shared transformer block recursed K times. At inference time the same weights become a draft model for self-speculative decoding — you already trained a cheap future-token predictor, you may as well use it.

## What we built in MegaCpp

### Mixture-of-Depths routing

The MoD implementation is large and carries its share of scar tissue. The core dispatch derives an effective routing policy from two generations of configuration flags. The legacy surface accepts `topk`, `threshold`, `progressive`, or `gateskip`; the newer surface splits that into a scoring function, a selector, a schedule, a target, an execution mode, and an attention-source axis for the attention-based scorer.

The three shipped routing modes:

- `topk` is the original MoD: a `Linear(D, 1)` scorer per layer, top-C% by score go through the block, the rest take a residual skip. Deterministic compute per layer. Requires a sort.
- `threshold` is the MoDification variant (Kulkarni et al. 2024, arXiv:2410.14268): no sort, just a per-token compare against a scalar threshold. Per-layer compute becomes adaptive — important layers pass more tokens, unimportant ones skip more. XLA-friendly because there is no dynamic top-k.
- `gateskip` is the simplest and currently the winner on our throughput-vs-loss frontier (GateSkip, arXiv:2510.13876). Every token goes through the block, then a soft sigmoid gate scales the block's contribution to the residual. An L2 sparsity loss on gate values pushes them toward zero, so "skipped" layers literally multiply the block output by something close to 0. No gather, no scatter, no top-k — just a learnable knob per token per layer.

The XLA path uses a compact `gather + FFN + gate + scatter` pipeline through JAX so XLA sees a `[C, D]` matmul rather than `[B*T, D]`, saving roughly `1/capacity_factor` FLOPs on the MXU. Gradients flow through `jax.vjp` inside a `torch.autograd.Function`. On non-XLA devices this silently falls back to the PyTorch implementation.

Two bugs from the ablation history matter here. First, MoD plus relation bias crashed: the wrapped block sees compacted `(B, T')` shapes but the relation bias path still expected `(B, T)` shapes, so broadcast failed. The fix was to skip relation bias on MoD-wrapped layers. Second, the attention-scorer bridge used to hardcode dense attention in a way that broke side-output accounting, and the full-block MoD path dropped the MoDA depth-KV contract. Both are explicit in MegaCpp now, with coverage reflected in the public examples and notes.

### MoDA depth-KV buffering

MoDA is intentionally small. `DepthKVBuffer` accumulates K/V tensors from each preceding layer — `_keys` and `_vals` lists detached from autograd (backward through 52 layers of concatenated K/V OOMs instantly, and MoDA is a context signal, not a gradient path). `.push(k, v)` at each layer adds a contribution, `.get(layer_idx)` concatenates layers `0..layer_idx-1` along the sequence dimension, yielding `(B, i*T, n_kv_head, head_dim)` depth K/V that gets concatenated with the standard sequence K/V before Flash Attention. Flash handles variable K/V lengths natively, so this is a "concat and dispatch" call, not a new kernel. Depth K/V does not have RoPE applied: cross-layer attention at the same position is position-invariant by construction.

For attention-bearing blocks we reuse the existing K/V projections — zero additional parameters. For non-attention blocks (e.g. the EBlock or MBlock variants) we add a lightweight `DepthKVProjection(D -> 2 * n_kv_head * head_dim, bias=False)` so those layers can still contribute depth K/V without a full QKV.

### Multi-Token Prediction and drafting

The MTP module implements the FastMTP and DeepSeek-V3 recursive shared-block design. Given `hidden_states` of shape `(B, T, C)` and a targets tensor, it predicts K future tokens with a single shared transformer block called K times. The trick that keeps it XLA-safe is the **roll-and-mask** pattern from MaxText: instead of slicing `hidden_states[:-k]` and `targets[k:]` at each depth, which would create variable-size tensors, we keep the full `(B, T, C)` shapes and use `torch.roll` to shift data left, plus a mask that sets the last `(k+1)` positions to `ignore_index=-1` so they do not contribute to the loss. Helper utilities handle the boundary housekeeping, including packed-document isolation.

At each depth `k` (0..K-1): roll ids and targets, embed the rolled token id, RMSNorm both the current hidden state and the rolled embedding, concat and project back to `C` through `self.proj`, run `self.block` (the shared weights), RMSNorm the output, and compute cross-entropy against the rolled targets using the shared `lm_head`. On CUDA we use `kernels.fused_linear_cross_entropy` to avoid materialising the full `(B*T, V)` logits. On XLA we fall back to `F.linear + F.cross_entropy` because the fused path's internal `.float()` upcast produces NaN gradients on XLA. Per-step losses are combined with exponentially decaying weights `alpha_k = beta^k / sum(beta^j)`; with `beta=1.0` they are uniform. The module is activation-checkpointed per block call during training and contributes nothing at inference.

One cast-and-preserve helper turned out to be load-bearing: Megatron tensor parallelism marks vocabulary shards with plain tensor attributes rather than a DTensor wrapper, and a naive bf16 cast inside MTP would drop them, which broke fused cross-entropy dispatch. The fix is to cast dtype while preserving the vocabulary-parallel markers.

The drafting path reuses the trained MTP module at inference time. The draft-model wrapper generates K draft tokens autoregressively: starting from the last verified hidden state and token ID, it fuses hidden state and token embedding, runs the shared block once, normalizes the output, computes logits against the shared LM head, samples or argmaxes the next draft token, and feeds that draft embedding back for the next step. This is a batched sequential loop, not the roll-and-mask pattern, because at inference there is nothing to mask.

The draft model is a fraction of the backbone's parameters (one shared block versus every layer), so drafting K tokens costs roughly `K / n_layers` of one main forward pass. Prerequisites for real speedup — KV cache rollback in the engine, acceptance sampling, and a benchmark that clears 1.0x on target hardware — are tracked separately.

## How it lands in production

The production port is shipped on a per-specialist basis because the three features have genuinely different cost profiles.

**MoD**: the public configuration surface provides fail-closed settings for both MoD and MoDA. MoD validation enforces that a layer list is provided, capacity stays in `(0, 1]`, auxiliary loss weight is non-negative, routing is one of the supported modes, and execution uses one of the supported backends. The `skip_first_n` knob, default 4, keeps the first few layers untouched because we observed routing instability when MoD was applied too close to the embeddings, and `skip_mamba`, default true, prevents the router from wrapping Mamba blocks, whose compaction contract does not compose cleanly with the SSM scan. The production feature surface is flag-driven: a specialist can enable `gateskip`, the cheapest and default mode, `threshold`, or `topk` as an explicit choice. `progressive` stays behind a more advanced surface and is not a default.

**MoDA**: the public configuration surface is intentionally minimal. The main contract is a depth-KV buffer that composes naturally with attention. It is off by default in shipped presets because the depth-KV buffer adds real memory to the attention call on long context, and at our target shapes the quality delta did not offset the memory cost. It stays available as a specialist feature flag.

**MTP**: the production training path uses a dedicated fast MTP layer rather than Megatron's default multi-token prediction block. The rewrites that mattered are Liger-Kernel fused linear cross-entropy, so there is no full-vocabulary logits tensor on the hot path, roll-and-mask static shapes so XLA and TorchDynamo see one graph, activation checkpointing inside the K-loop, one shared block reused K times rather than K distinct blocks, exponential-decay weighting with cadence, and step weights precomputed as buffers.

FastMTP is the path we train on in production. Megatron's default MTP block remains available as a fallback but is not the hot path.

The draft model and the self-speculative decode engine integration stay in the MegaCpp roadmap for v1. The draft math is correct; the blockers are on the inference-engine side: KV cache rollback, an acceptance kernel, and per-hardware benchmarking. They are being staged separately and are out of scope for this post.

## Ablations and what we kept

Snippets from the ablation history that shaped these decisions:

- `gateskip` sits on the throughput frontier and trains most stably. MoDA is measurable but expensive. Dense MTP costs a few percent per depth, so we typically run `K = 1` or `K = 3` by preset.
- MoDA at full depth-KV buffering is a real memory hit on long context. Detaching depth K/V from autograd is non-negotiable; backward through 52 layers of concatenated K/V OOMs immediately.
- MTP-as-loss is cheap once the lm-head-weight cast is correct. MTP-as-drafter is free at training time; the wall-clock win lives in the inference engine.
- The MoD wrapper rewrite that preserved dense-attention side outputs and re-plumbed the MoDA depth-KV contract was the single most consequential MoD fix of the cycle. Regression tests cover both.
- A NaN-scoring investigation on an H200-class GPU traced to the structure guard inside the dense-MTP path, not to MTP itself. Aligning the three dense-MTP variants with the same dense reference behavior fixed the Xid 31 fault and restored comparability between no-MTP, `MTP = 1`, and `MTP = 3` runs.

## Production checklist

- MoD must default to `gateskip` in shipped presets. `threshold` and `topk` are ablation modes, not defaults.

Three features, three axes of impact:

| Feature | What it skips or adds                | Default  | Dominant cost               |
|---------|--------------------------------------|----------|-----------------------------|
| MoD     | skips low-importance tokens per layer| gateskip | router + gather/mask path   |
| MoDA    | cross-depth K/V attention buffer     | off      | depth-KV memory on long ctx |
| MTP     | K-step future-token auxiliary loss   | K=1 or 3 | fused-CE + roll-and-mask    |

FastMTP uses a K-step roll-and-mask loop like this:

```python
# sketch
for k in range(K):
    h = shared_block(h)                   # one block recursed K times
    h_k = roll_and_mask(h, shift=k + 1)   # static-shape shift
    loss_k = liger_fused_linear_ce(h_k, lm_head_weight, labels_k)
    total += step_weights[k] * loss_k     # weights pre-computed as buffers
```

- `skip_first_n >= 4` and `skip_mamba=True` are invariants; overriding them is an explicit ablation.
- MoD-wrapped layers must not carry relation bias. The (B, T') vs (B, T) shape mismatch is a crash, not a quality regression.
- The MoD wrapper must preserve attention side outputs and the MoDA depth-KV side input. Regression tests exist for both.
- MoDA K/V must stay detached from autograd. If you ever need gradients through depth-KV, redesign the feature, do not just remove the detach.
- FastMTP requires bf16 LM-head weights with any vocabulary-parallel markers preserved across the cast. Treat the cast-and-preserve step as part of the feature contract rather than an incidental implementation detail.
- MTP on XLA must use the `F.linear + F.cross_entropy` path, not the fused CE kernel. The fused path's internal `.float()` upcast produces NaN gradients on XLA.
- MTP as drafter is not enabled in MegaCpp v1 inference. When it is, it needs KV cache rollback and an acceptance kernel, not just the drafter weights.

## References

- MegaCpp's MoD implementation centers on mixture-of-depths routing, XLA gather/scatter execution, and a fail-closed configuration surface.
- MegaCpp's MoDA implementation centers on a cross-layer depth-KV buffer and projection path for attention.
- MegaCpp's MTP implementation centers on roll-and-mask training, fused cross-entropy, and packed-document isolation.
- MegaCpp's speculative-decoding draft path remains a research-stack and is not enabled in MegaCpp v1 inference.
- The production fast-MTP path uses a dedicated multi-token prediction layer with Liger-fused cross-entropy.
- [Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models — Raposo et al., 2024]
- [MoDification: Mixture of Depths Made Easy — Kulkarni et al., arXiv 2410.14268]
- [GateSkip: Learnable Skip Connections for Efficient LLM Training — arXiv 2510.13876]
- [FastMTP — arXiv 2509.18362]
- [DeepSeek-V3 — arXiv 2412.19437]
