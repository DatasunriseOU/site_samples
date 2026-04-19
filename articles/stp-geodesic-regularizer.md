---
title: "Semantic Tube Prediction: the 10K-step gate, trajectory straightness, and the wiring mistakes that mattered"
description: "A grounded walkthrough of the STP-style auxiliary loss: the public sample, the multi-span and multi-layer variants, the 10K-step gate, and the integration mistakes that can quietly disable it."
date: "2026-04-18"
tags: ["stp", "geodesic", "regularization", "representation-learning", "training", "jepa"]
---

In this sample, Semantic Tube Prediction appears as a cheap auxiliary trajectory-straightness penalty on hidden-state trajectories. The STP loss sample selects ordered triples of positions, builds two local direction vectors, and minimizes `1 - cosine_similarity`. The math is intentionally small. The real engineering difficulty is making sure hidden states are collected on the actual training path, delaying the loss until after a warmup period, and keeping the multi-span and multi-layer variants shape-safe enough that the regularizer cannot silently disappear.

## Code and notes

- [Semantic Tube Prediction paper](https://arxiv.org/abs/2602.22617)
- [PyTorch `gather` documentation](https://pytorch.org/docs/stable/generated/torch.gather.html)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/)

Semantic Tube Prediction is easy to oversell if you start from theory alone. The paper frames a geodesic hypothesis for hidden-state trajectories, but the implemented sample is a much narrower straightness-style penalty. That is precisely why the code is valuable: it turns the idea into something humble and inspectable. The loss surface here does not introduce a second model, a contrastive queue, or a heavy prediction head. It samples triples `(s, r, t)` from hidden states, computes the vectors from `s -> r` and `r -> t`, and penalizes their angular disagreement. The result is small enough to keep around as an auxiliary term, but only if the integration stays honest.

That last condition matters more than it sounds. Several revisions broke the hidden-state collector in ways that left training alive while effectively disabling STP. The lesson from the current tree is therefore twofold: the loss itself is simple, and the wiring contract is the real risk surface.

## The core loss in the code sample is intentionally tiny

The key function is `compute_stp_loss(hidden_states, n_spans=1)`. It accepts either a single `(B, T, D)` tensor or a list of such tensors. The single-tensor path is the basic last-layer variant. The list path averages the per-layer losses and implements the multi-layer variant. Both routes end up at `_stp_loss_single`.

That helper does four things.

1. It exits early with zero loss when `T < 3`, because no ordered triple exists.
2. It samples three positions per batch element using static-shape integer operations, sorts them, adds offsets `[0, 1, 2]`, and clamps the result into the valid range.
3. It gathers `h_s`, `h_r`, and `h_t` and forms the direction vectors `d1 = h_r - h_s` and `d2 = h_t - h_r`.
4. It averages `1 - cosine_similarity(d1, d2)` across spans and batch elements.

The central lines are as compact as the idea promises:

```python
d1 = h_r - h_s
d2 = h_t - h_r
cos_sim = F.cosine_similarity(d1, d2, dim=-1)
return 1.0 - total_cos / n_spans
```

The module documentation also states the intended scope clearly. Variant A is one triple on the last layer. Variant B uses multiple triples per sample. Variant C averages the same loss across multiple selected layers. That is the right way to think about the feature set: there is one basic geometric penalty, and the variants trade off cost versus variance and depth coverage.

## Why the sampling strategy matters more than it first appears

A lot of regularizer code fails not because the loss is conceptually wrong, but because the sampling logic introduces hidden control-flow or shape instability. The STP sample takes the opposite approach. It samples a `(B, 3)` tensor of positions, sorts it, adds the `[0,1,2]` offset, and uses `gather` to read hidden states. That is exactly the kind of static-shape, branch-light pattern that public XLA guidance tends to favor.

That detail is more than a portability note. It explains why STP can remain a low-drama auxiliary term. There is no rejection loop repeatedly trying to find valid triples. There is no dynamic list of indices growing or shrinking with the data. The loss consumes a fixed set of tensor operations that scale with the hidden-state tensor you already have.

That design is exactly what you want in a regularizer that may be enabled in several training lanes. The more exotic the sampler becomes, the more likely it is that the “small extra loss” turns into a debugging magnet.

| Variant | Input to `compute_stp_loss` | Main tradeoff |
| --- | --- | --- |
| last-layer single-span | one `(B, T, D)` tensor, `n_spans=1` | cheapest, highest variance |
| last-layer multi-span | one `(B, T, D)` tensor, `n_spans>1` | lower-variance estimate, still cheap |
| multi-layer | list of `(B, T, D)` tensors | regularizes depthwise trajectory flow, higher integration burden |

The tests back this up. The note and sample surfaces show that more spans reduce estimate variance, that gradients flow through the loss, that zero or constant hidden states do not produce NaNs, and that the multi-layer version sends gradients to every layer in the provided list. Those tests are not proving the geodesic hypothesis in any philosophical sense. They are proving that the auxiliary loss behaves like a sane tensor program.

## The 10K-step gate is an optimization judgment, not ceremony

The blog summary for this topic mentions turning the regularizer on after 10K steps, and that choice is sensible when read against the code and experiment notes in this repo. STP is a smoothing bias on representational trajectories. Early in training, the model still has to carve out basic token geometry and task structure. Forcing local straightness too early can distort that process.

So the 10K-step gate should be read as an optimization policy: first let the base objective build a workable representational space, then add the curvature penalty once hidden-state trajectories are meaningful enough to regularize. This matches how auxiliary losses are usually safest in practice. They help most when they refine an existing manifold, not when they are asked to invent one from noise.

The important point is not whether 10K is a sacred universal threshold. It is that the project treats the gate as a real schedule parameter rather than a vague intuition. If you regularize too early, the problem is no longer just theory. You are changing the optimization landscape before the base model has stabilized.

## The integration risk is in the collector path, not in the cosine formula

The most instructive STP failures usually come from the integration path, not from the loss in the code sample. The failure mode is easy to understand: if the hidden-state collector sits under the wrong branch, or only records one subset of layers, the regularizer can quietly compute on the wrong tensor set or on nothing useful at all.

This is exactly the kind of bug auxiliary losses attract. Main training still runs. Tokens still flow. The step time may barely change. But the extra objective is effectively dead. That is why the correct engineering rule is stricter than “compute the loss somewhere.” The hidden-state collector must be unconditional along the relevant layer loop, and only the later loss-assembly logic should decide which collected states are consumed.

The public sample and its tests support that reading even though they exercise the loss module directly rather than a full model integration. The tests verify scalar shape, gradient flow, numerical stability, and multi-layer averaging. Those checks are necessary but not sufficient. The deeper integration contract is simple: if a model claims STP is enabled, the training path must actually populate the hidden-state list that the loss expects.

## Multi-layer STP is where notation and model structure start to matter

STP becomes more interesting once you place it in the hybrid model vocabulary used elsewhere in the project. The same stack that uses `A`, `M`, `E`, and `R` notation can expose hidden states after attention-family, Mamba-family, expert-family, and recurrent-tail blocks. That means a multi-layer STP run is not merely smoothing one homogeneous transformer depth. It is regularizing a mixed architectural path.

That is useful because hybrid stacks often accumulate representational bends at the boundaries between block families. Attention may rewrite token interaction patterns differently from a Mamba scan, and an expert block may introduce another change in geometry. A multi-layer STP loss can therefore be read as a mild consistency constraint across those transitions.

The glossary terms help here:

| Block family | Why STP might care |
| --- | --- |
| `ablock` / `A` | attention changes token-to-token mixing directly |
| `mblock` / `M` | Mamba-family scan layers change the sequence dynamics differently |
| `eblock` / `E` | expert routing can alter hidden-state geometry sharply |
| `rblock` / `R` | recurrent tails may have their own trajectory signature |

This does not mean STP is a bespoke hybrid-model trick. It means the auxiliary loss becomes more informative when the model is heterogeneous enough that “trajectory curvature” can reflect genuine architectural boundaries rather than just layer noise.

## The tests show what the project actually considers non-negotiable

The most reliable way to understand the intended STP contract is to look at what the tests insist on. The note and sample surfaces show several properties that are easy to overlook but highly practical.

- straight-line synthetic trajectories should drive the loss near zero
- random trajectories should produce a positive loss
- `T < 3` should return zero instead of crashing
- gradients should flow and should be nonzero
- multiple spans should lower estimator variance
- lists of layer tensors should propagate gradients to every layer
- bf16 and fp16 inputs should remain numerically safe

That collection is revealing. The project is not trying to prove an academic theorem about geodesics. It is enforcing the behavior of a production-adjacent auxiliary loss: cheap, differentiable, numerically stable, and structurally extensible.

A representative test snippet captures the expected invariant:

```python
layers = [torch.randn(2, 32, 64, requires_grad=True) for _ in range(4)]
loss = compute_stp_loss(layers, n_spans=4)
loss.backward()
```

If any layer in that list fails to receive gradients, the multi-layer contract is broken. That is the kind of grounded guarantee that matters more than broad claims about representation quality.

## Why this regularizer is cheap enough to keep, but only if the collector is trustworthy

The final judgment from the code is fairly balanced. This STP-style auxiliary loss is cheap enough to justify experimentation. It uses hidden states the model already produced. It adds a small gather-and-cosine computation. It has tests for gradients and low-precision dtypes. That makes it a plausible always-available auxiliary term once the base model has warmed up.

But the same code also shows why teams should be skeptical of “it’s enabled” as a status report. Auxiliary losses fail silently when their inputs are miswired. If the hidden-state collector lives under the wrong conditional, the loss still computes, but on incomplete or irrelevant states. In practice that means the math in the code sample is not the weak point. The weak point is the plumbing that feeds it.

So the grounded takeaway is simple. STP here is not a grand mystery. It is a small trajectory-straightness auxiliary loss with sensible variants, a delayed start, and a healthy test surface. The real discipline is ensuring that the model path populates the states it promises to regularize. Once that contract is solid, the loss is exactly what a useful auxiliary objective should be: modest, cheap, and hard to misinterpret.

## Further reading

- [Semantic Tube Prediction paper](https://arxiv.org/abs/2602.22617)
- [PyTorch `gather` documentation](https://pytorch.org/docs/stable/generated/torch.gather.html)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/)
