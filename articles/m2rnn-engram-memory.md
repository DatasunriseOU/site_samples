---
title: "M2RNN and Engram: The Memory Subsystem Inside the Hybrid"
description: "Where matrix-state RNN layers, causal n-gram Engram branches, and the learned concept bank fit inside our Mamba 3 + Transformer hybrid — and which pieces remain useful in the public memory stack."
date: "2026-04-18"
tags: ["m2rnn", "engram", "memory", "hybrid", "mamba3", "architecture"]
---

The hybrid stack described here is not "a transformer with some Mamba sprinkled in". It is a deliberate memory hierarchy: Mamba 3 SSM blocks for bulk long-range recurrence, attention blocks for sharp content-addressed lookups, M2RNN layers for matrix-valued recurrent state, an Engram branch that runs local causal n-gram features in parallel with the residual stream, and an optional concept-retrieval block that reads from a learned bank of global patterns. This post is about the three of those that are *not* vanilla attention or Mamba: M2RNN, Engram, and the concept bank.

## Why this setup matters

C++ training data has several memory demands that standard attention does not satisfy cheaply. Long files need linear-in-length recurrence — Mamba handles that. Local n-gram patterns (operator idioms, template boilerplate, four-token sequences that are effectively copied everywhere) want a cheap causal smoother, not a full attention head. Cross-document abstractions ("this is a linked-list insert regardless of tokens") want a global pattern bank you can cross-attend to without paying `T x T`. M2RNN occupies a different niche again: a matrix-valued recurrent state that can store more per step than a vector SSM without breaking into full attention.

None of these are critical-path on their own. What matters is that layering them is cheap — the features are additive to the residual and most of them are zero-initialised — so ablations are clean, and the wins stack.

## What this memory stack includes

### M2RNN

The research M2RNN module implements the layer. The core recurrence is straightforward:

```
h_t = tanh(h_{t-1} @ W + k_t ⊗ v_t)
h_t = f_t * h_{t-1} + (1 - f_t) * h_t
y_t = q_t @ h_t
```

The state `h` is a matrix, not a vector — at head dims `K=64`, `V=16` the per-head state is 64x16. That extra capacity versus a vector SSM is the whole point. `_softplus_decay_gate` produces the forget factor `f` in SSM style (`exp(-A * softplus(x + dt_bias))`), and the per-layer setup is `input_proj` then optional `causal_conv1d`, then split into `q`, `k`, `v` plus the decay and output gates, then recurrence, residual `v*D`, output gate, norm, `output_proj`.

The file has three forward paths chosen at import time. The default is the XMA Triton kernel (`_xma_m2rnn_forward`) when `xma.functional.m2rnn` is importable. If not, we fall back to a pure-PyTorch sequential loop (`_torch_m2rnn_forward`) wrapped with `@torch.compiler.disable` so dynamo does not try to unroll the step loop and explode the graph. The wrapper reads `MEGACPP_STARTUP_TRACE` once at import (a dynamo-friendly constant) instead of polling `os.environ` every step. These are the kinds of things that matter more than the math when you want `regional_compile` to stay stable across epochs.

The Megatron bridge around that layer is the glue for training inside a Megatron stack. The production bridge adapts the M2RNN layer to Megatron's mixer protocol: it accepts the standard module-construction arguments, transposes from `[seq, batch, hidden]` to `[batch, seq, hidden]` for the forward path, returns `(output, None)` so the surrounding residual path stays correct, and can run the pure-PyTorch recurrence rather than the kernel when compile compatibility matters. A small config bridge reconciles the earlier attribute names with Megatron's transformer configuration when both exist.

### Engram

The local causal n-gram branch runs in parallel with the main block: given `(B, T, C)` hidden states it produces `(B, T, C)` features that get added back into the residual. The original mode (`gated=False`, `conv_kernel=0`) is the minimal version — project to a bottleneck, compute causal local averages at orders 2/3/4 via `avg_pool1d`, mix per-order, project back out, zero-init. The upgraded mode (`gated=True`, `conv_kernel=4`) adds two things on top. First, DeepSeek-style context-aware gating: `alpha = sigmoid(RMSNorm(h)ᵀ · RMSNorm(k) / sqrt(d))`, where `h` is the pre-norm residual and `k` is the linearly projected n-gram features. Second, a grouped causal convolution with SiLU activation on top. Two conv implementations are selectable: `maxtext_depthwise` uses `nn.Conv1d(groups=D)` on CUDA and unrolls by hand on XLA, while `xla_safe` uses a manual `unfold`-and-sum on CUDA and the same manual loop on XLA. The per-device split exists because the fast CUDA conv path is not well optimised on TPU.

There is subtle work in `_same_doc_shift_mask` and `_causal_local_average`: when documents are packed together, the causal smoother must not leak across document boundaries. Engram threads `doc_ids` through every shift so an n-gram window that would cross a document boundary is treated as left-pad zeros. This was a real regression in earlier runs — packing was correct but Engram was quietly mixing documents, and the attention validity tests did not cover it until we added them.

The companion `_RMSNorm` class is deliberately small: no learnable parameters, and the forward is a single `F.rms_norm` call with `weight=None`. An earlier version did `x.pow(2).mean(-1)` and a manual `.float()` upcast. The manual path broke fusion under Inductor (the explicit dtype change was a fusion boundary) and overflowed in bf16 for large activations. The fix both stabilised bf16 training and reclaimed throughput.

### The engram "package" and concepts

The brief asks about an `engram/` package, and it is worth being honest: the Engram path discussed here is a single module rather than a larger package. The broader "engram concept" — external learned memory the model reads from without writing — is represented in two places. The local half is `EngramBranch` above. The global half is the concept-retrieval module. `CBlock` is a cross-attention block where the queries come from hidden states and the keys/values come from a learned `Embedding(n_concepts, concept_dim)` — the concept bank. There is no causal mask because the concepts are global prototypes, no RoPE on the concept K/V because the concepts are order-invariant, softmax is done in fp32 for stability, and the output projection is zero-initialised so dropping a `CBlock` into any layer position is identity at step 0. Concepts are read-only: the bank is learned by gradient descent, never written during the forward pass. That is the distinction we care about — Engram is the local reader, the concept bank is the global reader, and neither is a write-enabled episodic memory in this shipment.

## How it lands in the public sample

The production layer keeps the pieces that paid off and drops the more speculative surfaces.

### M2RNN in the public sample

The production M2RNN seam lives in the Megatron spec layer, plus a public configuration sample. That configuration holds `d_model`, `k_head_dim` (default 64), `v_head_dim` (default 16), `conv_kernel` (default 4), gradient clipping, a residual flag, and `A`/`dt` init ranges; a single builder pulls those from the Megatron config with sensible defaults. The big rewrite is the kernel path: the public M2RNN Triton sample provides a Triton M2RNN scan (`m2rnn_scan_triton`) that is a drop-in replacement for `_torch_m2rnn_forward`. On our reference hybrid geometries (`B=2, S=4096, H=8, K=64, V=16`) the Triton path is dramatically faster than the Python reference loop — the exact multiplier depends on hardware, but the reference loop is orders of magnitude slower and is used only as a deliberate debug path via `CPPMEGA_M2RNN_KERNEL=torch`. If Triton is not importable, the wrapper warns loudly rather than silently degrading: running the reference loop in production is not something we want to discover from a throughput dashboard.

The config bridge between the earlier M2RNN configuration surface and the Megatron transformer configuration is intentionally simple: there is a single entry point, and M2RNN reads its fields directly from the Megatron config via stable attribute access rather than through an extra shim layer.

### Engram in practice

the public Engram config sample defines `EngramConfig` and `NgramHashConfig`, both fail-closed. `EngramConfig.from_args` validates layer indices, n-gram orders, bottleneck dim, dropout, conv kernel, and conv impl (must be `"xla_safe"` or `"maxtext_depthwise"`). The `EngramBranch` implementation itself is what matters architecturally: a small causal n-gram memory branch with doc-id threading, a fused RMSNorm path, and explicit convolution-mode selection. `NgramHashConfig` is adjacent rather than identical: it handles hashed n-gram token embeddings, not the main Engram residual branch.

### Concepts and `CBlock`

The concept bank (`CBlock`) is easiest to describe as a cross-document retrieval block. In the public glossary, `cblock` means a lightweight coordination or connector block; in this memory-stack context it is the optional concept-retrieval tier rather than the default path. The reason it stays optional is pragmatic: in ablation runs the concept bank was additive but small, and it is parameter-heavy at useful `n_concepts`. The design remains worth keeping because the underlying idea has strong precedents in [Memorizing Transformers](https://arxiv.org/abs/2203.08913) and [Flamingo](https://arxiv.org/abs/2204.14198). For the memory stack discussed here, the default tiers are M2RNN + Engram + Mamba + attention, not five always-on subsystems.

## Ablations and what we kept

The published notes and companion articles tell a consistent story about what survives contact with real hardware:

Memory-subsystem roles at a glance:

| Module       | Role                                  | Cost profile                 | Default |
|--------------|---------------------------------------|------------------------------|---------|
| Mamba 3      | bulk long-range recurrence            | O(N), low bandwidth          | on      |
| Attention    | sharp content lookups                 | O(N^2) on its minority share | on      |
| M2RNN        | matrix-valued recurrent state         | matrix state per head        | on      |
| Engram       | local causal n-gram smoother          | cheap, additive residual     | on      |
| CBlock       | cross-doc concept retrieval           | parameter-heavy              | off     |

- M2RNN kernel path dominates. The pure-PyTorch loop is a correctness reference, not a training path. The XMA/Triton kernels are the only sensible choice once the model is past toy size.
- Engram gated + conv=4 is the default. Earlier variants without conv were fine but measurably weaker on code benchmarks. The conv adds real capacity; the gate prevents it from drowning out the main block.
- `F.rms_norm` fused path in Engram's `_RMSNorm` is not optional. The manual bf16 variance path overflows on real activations and breaks Inductor fusion. This is the single largest regression we fixed in the Engram subsystem.
- Cross-document bleed in Engram's conv and n-gram pools is a correctness bug, not a quality one. Packed sequences + Engram without `doc_ids` threading silently mixes documents. Regression tests for cross-document isolation now cover both the n-gram pool and the conv kernel.
- mHC (multi-stream memory hop connections) sits on top of Engram. A bug we found during review: mHC without Engram layers was silently no-op because the mHC layer list fell back to the empty engram layer list. It now defaults to all layers with a warning when `--mhc` is used without an explicit engram layer set.
- The full stack (Mamba + Engram + mHC + MTP + MoD + structure + ngram hash) is measurable in the throughput tables: baseline MoE + Mamba + Engram on the training bench is our reference, adding Engram + mHC costs memory bandwidth but adds capacity on code tasks.
- M2RNN + `regional_compile` works, M2RNN + whole-model compile does not without the `@torch.compiler.disable` break. Keeping the Triton call outside the compiled graph is non-negotiable.

## Production checklist

- The production M2RNN config should be built through the canonical builder. Any direct construction must still pass all fields; the dataclass is frozen.
- `CPPMEGA_M2RNN_KERNEL` is a debug knob. Never set it to `torch` in a real training run; if Triton fails to import, fix the environment before starting.
- Engram's `conv_impl` must match the training device: `maxtext_depthwise` on CUDA, `xla_safe` on TPU. Crossing them trains correctly but loses throughput.
- Packed sequences + Engram require `doc_ids` threaded through every branch. Without it, Engram leaks across document boundaries.
- `_RMSNorm` inside Engram must use `F.rms_norm(x, (D,), weight=None, eps=eps)`. Do not reintroduce the manual `x.pow(2).mean(-1)` variance path.
- When enabling mHC, always specify engram layer indices explicitly. Relying on the default fallback list is a known silent-noop trap.
- Concept bank (`CBlock`) is off by default in the public memory-stack recipe described here. Turning it on is an ablation, not the baseline configuration.
- Keep M2RNN outside whole-model compile. `regional_compile` only, with the step loop under `@torch.compiler.disable`.

## References

- [Model glossary](https://megacpp.com/blog/megacpp-model-glossary/)
- [Hybrid layout notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md)
- [Tensor parallel and sharding article](https://megacpp.com/blog/tensor-parallel-and-sharding/)
- [Mamba-3 paper](https://arxiv.org/abs/2603.15569)
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [Flamingo](https://arxiv.org/abs/2204.14198)
- [OpenLM engine](https://github.com/mlfoundations/open_lm)
