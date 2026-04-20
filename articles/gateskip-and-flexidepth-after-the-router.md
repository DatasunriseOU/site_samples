---
title: "GateSkip and FlexiDepth after the router"
description: "How MegaCpp treats dynamic-depth features as bookkeeping and wiring problems after the router, not just as a paper-level skipping idea."
date: "2026-04-19"
tags: ["routing", "dynamic-depth", "gateskip", "flexidepth", "core-blocks"]
---

The interesting part of dynamic depth is not the first router score. The
interesting part is everything that comes after it: residual semantics, loss
bookkeeping, frozen-adapter wiring, and a reliable accounting of which tokens
actually used which layers.

That is where the MegaCpp examples are strongest. They do not stop at the idea
of skipping compute. They show what the runtime still has to preserve after the
router has already made its decision.

## The router is only the start of the contract

GateSkip and FlexiDepth both sound simple when reduced to one sentence. Route a
token, skip a layer, save compute. But the local examples make the real cost
visible.

For GateSkip, the skip decision still has to live inside a residual stream that
remains well-defined when some tokens do less work than others. That is why the
examples split the surface into:

- a residual router example
- explicit loss bookkeeping
- block-taxonomy context so the skip decision stays tied to a real block family

For FlexiDepth, the examples go farther. They preserve not only the skip logic
but also the adapter surface and the frozen-backbone wiring. That is the right
public lesson: once dynamic depth is attached to a pretrained or partially
frozen stack, the question is no longer just "which layer was skipped?" It is
also "which moving part is still allowed to learn?"

## Why MegaCpp keeps these surfaces separate

The local split between GateSkip and FlexiDepth is useful because the two ideas
pay different operational costs.

GateSkip in this pack is primarily about token-wise gating and the accounting
that follows from it. The bookkeeping sample matters because sparsity pressure
is easy to describe badly. If the runtime cannot show how the gate loss, budget
pressure, and actual token path line up, then the feature is only half real.

FlexiDepth is more structural. The examples preserve layer-usage stats,
adapter-side movement, and a frozen-backbone story. That makes FlexiDepth less
like a routing paper and more like a controlled migration path for dynamic
depth on top of an existing model.

## Why this belongs in core blocks rather than in a generic routing folder

These examples live next to Engram, mHC, n-gram embeddings, and block taxonomy
for a reason. In MegaCpp, routing is not treated as a free-floating policy
module. It is attached to real block families and real residual paths.

That is important because a skip surface can interact badly with branch mixing
or residual alternatives if the runtime pretends they are independent. The
residual-path and mHC-adjacent examples make that risk explicit. Dynamic depth
is not only a router problem. It is a stream-integrity problem.

## What the public examples prove

The useful claim is narrower than "we support dynamic depth."

The examples prove that MegaCpp has a public-safe contract for:

- token-wise residual gating
- skip-loss and usage accounting
- frozen-backbone adapter wiring for dynamic-depth variants
- block-family-aware placement of these features in a larger hybrid model

That is enough to support a serious architectural claim. It shows the feature
exists as a runtime surface rather than as a research aspiration.

## Prior art and context

The general idea is not unique. Mixture-of-Depths is the clearest direct prior
art for dynamic token-wise depth allocation. FlexiDepth-style work extends that
idea toward pretrained-model adaptation, while older adaptive-compute papers
such as ACT, PonderNet, Universal Transformers, and Depth-Adaptive Transformer
show the longer history of learned variable compute. GateSkip sits closer to
residual-gated layer skipping. MegaCpp's local contribution is narrower and more
practical: the public examples show how these ideas survive contact with block
taxonomy, residual contracts, adapter wiring, and training-time bookkeeping.

## References

- [GateSkip residual router sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/core_blocks/gateskip_residual_router_sample.py)
- [GateSkip loss bookkeeping sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/core_blocks/gateskip_loss_bookkeeping_sample.py)
- [FlexiDepth adapter sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/core_blocks/flexidepth_adapter_sample.py)
- [FlexiDepth loss stats sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/core_blocks/flexidepth_loss_stats_sample.py)
- [FlexiDepth frozen adapter wiring sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/core_blocks/flexidepth_frozen_adapter_wiring_sample.py)
- [Residual paths sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/core_blocks/residual_paths_sample.py)
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)
- [FlexiDepth](https://arxiv.org/abs/2503.23798)
- [GateSkip](https://arxiv.org/abs/2510.13876)
- [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
