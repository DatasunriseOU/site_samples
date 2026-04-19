---
title: "Long context and attention sinks: what actually held up past 16K"
description: "YaRN, RNoPE, packed-document masking, attention sinks, massive activations, and query-dependent output gating: a field report on which long-context techniques survived contact with the MegaCpp C++ corpus."
date: 2026-04-18
author: "David Gornshtein"
tags: ["long-context", "yarn", "rnope", "attention-sinks", "gated-attention", "rope", "c++"]
---

# Long Context and Attention Sinks: What Actually Held Up Past 16K

The MegaCpp specialists have to read repositories, not snippets. A realistic training sample from our v4 context-graph sampler is a `Callers -> Target -> Callees` bundle that reaches 64K tokens of C++ before the budget cuts in; a realistic inference sample is a translation unit plus its headers and a stack trace, which comfortably exceeds 32K. That puts us in the bucket of "every long-context paper applies in principle, none of them apply cleanly". This post is a field report on which long-context techniques survived contact with the C++ corpus, what broke in ways that did not show up on the short-context ablations, and which mitigations we actually kept past 16K.

## The two axes we care about

Long-context work on our stack breaks into two non-overlapping axes that are easy to confuse. The first is **positional extrapolation**: the model was pretrained at one sequence length and we want it to behave reasonably at a much longer one. YaRN, RNoPE, RoPE theta scaling, and a context-length curriculum all live on this axis. The second is **numerical stability at long range**: attention sinks, massive activations, first-token bias, packed-document prefix bias. These are properties of what the model actually computes once the positions are correct, and they are orthogonal to whether the positional encoding extrapolates. We hit both axes in the same training cycle and it took us a while to stop treating them as the same problem.

## Positional extrapolation: YaRN, RNoPE, and the parts that worked

The design target is 128K tokens today with a clear path to 1M. The FLOP math dictates that we cannot just pretrain at 1M. A single 1M-token sample through a 4B model is enough to stall any single-node training budget for a week, and the KV-cache alone at 1M tokens and standard GQA is comfortably north of 130 GiB before MLA compression. The industry pattern is to pretrain short and extend through a staged curriculum, and that is what we do.

The concrete schedule on the specialists is four stages: 8K pretrain (the bulk of tokens), then LCFT-1 at 32K (YaRN scale 4), LCFT-2 at 128K (YaRN scale 16), and LCFT-3 at 512K (YaRN scale 64). Each stage is short relative to the pretrain budget (roughly 140h : 8h : 8h : 4h of wall-clock for the 4B backbone), and each stage consumes training samples built from concatenated v4 context graphs sized to the stage's target length.

YaRN is the positional scaling we kept. The core of the algorithm is the frequency-band split: the high RoPE frequencies encode local position and extrapolate cleanly, the low frequencies encode global position and must be interpolated down when the max sequence length grows, and the middle band is smoothed. `yarn_find_correction_range` computes the two band edges from `beta_fast` and `beta_slow`, and `yarn_get_mscale` returns `0.1 * log(scale) + 1.0` for `scale > 1`, which is the attention-logit rescaling that keeps softmax behaviour consistent with the original training temperature after the frequencies change. This is plain RoPE with a frequency-aware twist, it is checkpoint-compatible with the short-context pretrain (the weights are unchanged, only the positional embedding computation changes), and it does not require a separate training phase to activate. It does require the context-length curriculum to activate well, which is the next-biggest lesson from this cycle.

RNoPE (the hybrid RoPE/NoPE scheme where a fraction of the heads do not receive a positional encoding at all) is on our roadmap but not in the specialists as shipped. The short-context ablations showed that RNoPE interacts badly with our packed-document training regime until intra-document masking is correct, and we fixed the masking first. The non-positional heads are supposed to provide a positionless "retrieval" path that the YaRN-scaled positional heads cannot; without clean document masking, those heads learn spurious cross-document retrievals instead. Once masking is solid the RNoPE ablation becomes meaningful, and that is the next step on the extrapolation axis.

The three things that actually moved long-context accuracy past 16K were not glamorous. They were: fixing intra-document masking, switching from `best_fit` packing to a packing policy that does not crop document prefixes, and extending the context curriculum to actually reach the target length. YaRN by itself on the pretrain checkpoint did not clear any of our long-context evals; YaRN on top of a correctly curriculum-extended model did.

## Packed-document masking, which turned out to be load-bearing

The short version: a transformer trained on packed sequences that ignores document boundaries learns to attend across documents, and a transformer that does that looks fine at 4K and degrades badly at 32K. The long version is worse, because our original packing policy was `best_fit`: pack the largest document first, fill remaining slack with the next-largest that fits, repeat. That policy systematically crops document prefixes: the first 4K tokens of a 10K-token document land in one packed row, and the remaining 6K either get cropped or land in a different row. A model trained on that distribution oversamples document starts and undertrains document interiors, which is exactly the kind of bias that surfaces as "the model has a weird affinity for line 1 of any file" at eval time.

The fix is two changes. First, an intra-document additive mask that blocks attention from a token in one packed document to tokens in any other document in the same row. This is straightforward to build from a per-row `doc_ids` tensor and applies cleanly to FA3, FlexAttention, and the manual SDPA fallback; we carry the mask through the same sequence-parallel path as the causal mask. Second, a packing policy that respects document boundaries and does not crop prefixes: either bin-packing variants that take the full document as an atomic unit and accept the resulting packing inefficiency, or a continuous-packing scheme with explicit boundary markers that the masking layer reads. We run the latter in production. The document-level eval metric that was silently regressing on `best_fit` recovered as soon as prefixes stopped getting cropped; the long-context eval that had been blamed on YaRN turned out to be a masking bug in disguise.

## Attention sinks, massive activations, and the gated-attention RFC

The other axis is numerical. The short version of the phenomenon, from the spike/sparse/sink literature, is that pre-norm transformers trained on long sequences develop two related pathologies. The first is the **attention sink**: a few tokens (often the first token) receive a disproportionate fraction of attention mass, regardless of content, as a kind of "null attention" valve. The second is **massive activations**: a few hidden units in a few tokens grow to magnitudes orders of magnitude larger than the rest of the tensor, which is the activation-space counterpart of the sink behaviour in attention-space. They are related but not identical; fixing one does not automatically fix the other.

The mitigations we evaluated fall into four buckets.

**Static controls** (already in mainline): `qk_norm`, `qk_clip_threshold` with an epsilon, the `attn_softcap` on attention logits, and the `output_softcap` on the LM head. These are cheap and they compose with everything else. `qk_norm` is the one we are explicitly careful about: the spike paper is fine with it but the long-context ablations in our own code showed that `qk_norm` interacts with long-range positional retrieval, and we keep it off on heads whose long-context eval suffers from it. Softcaps stay on everywhere; they are a clean guard against logit explosion and they compose with FA3 through the kernel's `softcap` argument.

**Streaming sinks** (StreamingLLM-style): retain the first K "sink" tokens in the KV-cache forever plus a sliding window of the most recent tokens. This bounds decode memory growth and is useful as a serving heuristic. It is not a substitute for long-range recall; pinning tokens 0..3 plus the last 2048 does not let the model answer a question whose answer lives at token 8000. For our packed documents the sink policy also has to be document-relative, not row-global: a sink token in the middle of a row from the previous document is not a sink for the current document. We kept the document-relative variant for serving only, and we explicitly do not claim it as a long-context solution.

**Post-attention output gating** (from the Qwen gated-attention paper). This is the mitigation we moved first on. A query-dependent sigmoid gate multiplies the attention output before the `c_proj` projection; the gate is computed from the query state, stays outside the attention kernel contract, and is checkpoint-compatible because it collapses to identity when the gate weights are initialised to produce ~1. It addresses sinks more directly than pure bounded squashing, it applies uniformly to dense and sparse attention paths (including our DSA indexer), and it does not require any kernel changes. The decision in our attention-sink-mitigation RFC was to ship this as the first-line mitigation and measure before touching anything else.

**DynamicTanh** (from the "Transformers without Normalization" paper). Tempting because it simultaneously addresses massive activations and removes a whole class of norm layers, but the blast radius is enormous: normalization replacement changes full training dynamics, not just attention routing, and initialisation sensitivity is real. We kept it in a separate research track. If gated attention closes the measured gap on first-token mass and outlier percentiles, DynamicTanh becomes unnecessary; if it does not, DynamicTanh is the next thing to try, not the first.

The instrumentation that forced these decisions into an order was the other surviving piece of the RFC. Before any mitigation ships, we measure: first-token attention mass on a fixed eval prompt set, max and high-percentile hidden activations per layer, prefix-versus-suffix attention usage on packed documents, and sink behaviour per document rather than per packed row. A mitigation that does not move at least one of those numbers is not a mitigation, it is a refactor, and we do not ship it as the former.

## What actually held up past 16K

On the 4B specialist backbone evaluated at 4K / 16K / 32K / 64K on a code-reasoning harness built from repository-level context graphs:

Below 16K, nothing we changed moved the number meaningfully - well-behaved RoPE and reasonable packing already cover that range. At 16K the intra-document masking fix and the packing policy change were the dominant improvements: the model stopped spuriously attending across concatenated training documents, document-interior accuracy recovered, and a family of cross-translation-unit hallucinations dropped out. YaRN was present but not load-bearing here.

At 32K and 64K, YaRN with the correct frequency-band correction was load-bearing. Without it, precision on long-range signature matching ("the exact signature of `Buffer::append` declared 30K tokens ago") collapsed across every eval we had. With it, long-context tracked the 16K eval within single-digit points, the best result we have on this axis. Output gating was net positive at this length; the outlier-channel mass moved off the first token and onto the gate.

At 128K the picture is less complete. MLA weight absorption and paged KV cache make 128K tractable on a single B200 for serving; whether the model actually uses the full window for retrieval, versus relying on the streaming-sink slice plus recent context, is a question the current eval set does not cleanly separate. A middle-of-context retrieval eval is on the roadmap.

We do not claim streaming sinks plus a recent window are a long-context solution; they are a decode-memory heuristic that composes with everything above. We also do not claim any of this transfers to the Mamba-3 M-blocks, which have their own recurrence dynamics; this post is about the minority of attention blocks (MLA and DSA) in the hybrid stack.


## Long-context techniques: what held up

| Technique | Window | Status on the C++ corpus | Notes |
|---|---|---|---|
| YaRN rope scaling | up to 32K | kept | clean extrapolation past 16K once `attn_factor` is tuned |
| RNoPE (rotary skip-band) | 16K-32K | kept on a subset of layers | helps long-doc recall, costs a few MFU points |
| Packed-document masking | any | kept | `cu_seqlens` boundaries, document-aware attention |
| Attention sinks (first 4 tokens) | any | kept on inference path | lower training-time perplexity drift past 8K |
| Massive activations (handful of channels) | any | tracked, not actively suppressed | telemetry only; intervention regressed loss |
| Query-dependent output gating | any | kept | small win, composes with gated attention |

The packed-mask shape we feed Flash Attention:

```python
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len, max_seqlen_k=max_len,
    causal=True, softcap=softcap_value,
)
```

## References

- `06-long-context.md`
- `12-attention-sink-mitigation-rfc.md`
- `TEMPORAL_CODE_DYNAMICS_ROADMAP.md`
- `deepseek_mla_strategy.md`
- `training_review.md`
- `architecture_and_eval_en.md`
