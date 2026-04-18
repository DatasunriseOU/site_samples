---
title: "Geodesic loss design notes: span sampling, layer choices, and XLA-safe constraints"
description: "How the STP geodesic auxiliary loss is implemented in the research stack, why it samples ordered triples instead of predicting future latents, and what the MegaCpp landing should preserve."
date: "2026-04-18"
tags: ["design", "stp", "geodesic", "xla", "representation-learning"]
---

The geodesic auxiliary objective in the research repo is small enough to be underestimated. It is not a big subsystem, not a new decoder head, and not a speculative latent-prediction tower. It is a narrow regularizer in the public STP loss sample that asks a more modest question: if hidden states form a trajectory through representation space, do short local segments stay roughly straight? That choice matters because it keeps the feature cheap, backend-friendly, and easy to gate from the training loop.

**TL;DR:** the implemented STP design samples ordered triples `(s, r, t)` from existing hidden states and penalizes curvature with `1 - cos(h[t] - h[r], h[r] - h[s])`. The design deliberately avoids predictor heads, data-dependent control flow, and shape polymorphism. The good part is that it stays easy to wire into the runtime contract already used by the prototype. The still-open part is policy: how many spans to sample, which layers to supervise, and when to turn the loss on during training.

## What the actual objective does

the public STP loss sample in the research repo defines the geodesic loss directly. The module documentation describes the hypothesis in plain terms: hidden-state trajectories are assumed to be locally linear, so the loss should punish curvature rather than predict the next state explicitly. The implementation exposes one entry point, `compute_stp_loss`, which accepts either a single hidden-state tensor `(B, T, D)` or a list of such tensors for the multi-layer variant.

The scalar objective is simple:

```text
L_STP = 1 - cos(h[t] - h[r], h[r] - h[s])
```

That formula tells you almost everything about the intended behavior.

| Design choice | What the code does | Why it matters |
| --- | --- | --- |
| Geometry target | compares consecutive direction vectors | regularizes local straightness instead of predicting a future latent |
| Sampling unit | ordered triples `(s, r, t)` | ensures there is an intermediate point and therefore a notion of curvature |
| Core math | `gather`, subtract, cosine similarity | keeps the kernel narrow and easy to transport across backends |
| Layer surface | single tensor or explicit list of tensors | allows final-layer-only or selected-layer supervision |

There is no learned predictor head in the implementation. There is no teacher model. There is no extra projector whose hidden cost later pollutes memory accounting. That absence is not an omission. It is the point. The design tries to get a useful geometry prior while staying close to the exact hidden-state surfaces the training loop already owns.

The three documented variants are also explicit in the module documentation: Variant A is one triple on the last layer, Variant B is `N` triples on the last layer, and Variant C averages the same loss across a selected list of layers. That structure is narrow enough to reason about and broad enough to support real ablations.

## Why span sampling looks the way it does

The most important code in `_stp_loss_single` is not the cosine itself. It is the sampling logic. For each span, the function samples a `(B, 3)` tensor of base indices with `torch.randint`, sorts along the last axis, adds offsets `[0, 1, 2]`, and clamps to the valid sequence range. That sequence is what gives the feature its implementation character.

First, it guarantees ordered positions without host-side conditional logic. The function does not branch on special cases per sample. It builds a uniform tensor-shaped path that works the same way for every batch element. That is why the module can credibly describe its operations as XLA-safe.

Second, it keeps the estimator local. The loss is not asking whether the first token and the last token lie on a globally straight manifold path. It checks whether short consecutive segments align. In practice that makes the objective easier to interpret: it is a local curvature penalty, not a whole-sequence reconstruction signal.

Third, the cost scales with `n_spans`, not with vocabulary size, sequence-wide pairwise comparisons, or a separate prediction head. The docstring even calls the operation budget out as roughly three gathers, two subtractions, and one cosine per triple. The exact FLOP wording is informal, but the engineering intent is precise: this feature is supposed to be cheap enough to survive contact with real training.

The tests in sanitized public test excerpts reinforce that design. They verify scalar output, the `[0, 2]` loss range implied by `1 - cos`, near-zero loss on a synthetic straight-line trajectory, positive loss on random trajectories, correct behavior for short sequences, and gradient flow. That is a solid unit-level contract for a regularizer. It means the code is not just mathematically plausible; its edge conditions are intentionally covered.

## Layer selection is the real policy surface

The math kernel is simple. The real design question is where and when to apply it.

The STP implementation itself supports either a single hidden-state tensor or a list of tensors. In the multi-layer case, `compute_stp_loss` computes one scalar per layer, stacks them, sums them, and divides by the number of layers. That averaging rule matters because it rejects an implicit “all layers by default” story. The feature expects explicit layer selection.

That aligns with the broader training contract in the prototype. Public training runtime notes show that the STP coefficient is logged explicitly, warn that pipeline-parallel training drops auxiliary losses including STP, and gate activation with a delayed start policy. So the system is already split into two pieces:

1. the public STP loss sample answers how geodesic curvature is measured.
2. the training runtime answers when STP participates and how strongly it is weighted.

That separation is the right one to preserve in MegaCpp. Auxiliary objectives become hard to maintain when training policy leaks into the math kernel. Here the current arrangement is healthier: `compute_stp_loss` stays reusable, while step gating and feature enablement remain runtime concerns.

One practical implication is that any future default should stay explicit. If MegaCpp promotes this feature, a preset should state whether STP is applied to the last layer only, to a curated list of intermediate layers, or to some architecture-specific slice. It should not silently guess.

## Why the XLA-safe claim is not just marketing

The module documentation says “All operations are XLA-safe: static shapes, no data-dependent branching.” That line is easy to wave away unless you read it next to the TPU docs and the wider runtime code.

the public TPU setup note is very explicit that the TPU lane values static compiled graphs, per-micro-step compilation boundaries, and predictable shape behavior. The current TPU contract is narrower than a generic “just use torch on TPU” story. The runtime disables model `torch.compile(...)` on TPU, uses `torch_xla.compile()` around forward and backward by micro-step, and treats changing shapes or host-driven scalar behavior as regressions. In that environment, a regularizer that introduces dynamic control flow or variable output structure would be expensive even if its math looked elegant.

STP avoids that trap.

| Runtime concern | STP posture |
| --- | --- |
| changing tensor ranks | none |
| predictor-head materialization | none |
| host-driven conditionals in the kernel | none |
| auxiliary outputs with irregular structure | none |

That is the part worth carrying forward. A geodesic objective is only operationally useful if it respects the same graph-stability rules as the rest of the training stack. The current implementation does.

There is one caveat: “XLA-safe” does not mean “free.” If the training loop has to collect multiple layer activations solely for STP, that collection cost is real. But that cost is visible and policy-controlled. It is not hidden inside a second prediction tower or a backend-hostile control path.

## How this maps into MegaCpp

MegaCpp already has the right habits for introducing narrowly scoped runtime features. You can see that style in multiple places: public Mamba spec samples make layer-stack composition explicit, and public run-detail surfaces expose auxiliary-loss weights such as the STP coefficient rather than hiding them in opaque presets. STP should land the same way.

The likely stable shape is:

```yaml
stp_enabled: true
stp_weight: 0.02
start_step: 1000
stp_n_spans: 4
stp_layers: "4,8,12"
```

That sort of config expresses the policy cleanly. It says when STP begins, how much estimator stability is purchased with extra spans, and which representation surfaces receive the geometric bias.

The main things not to do are just as important.

| Bad landing choice | Why it is wrong |
| --- | --- |
| hiding STP behind an architecture-specific heuristic | makes comparisons impossible across runs |
| automatically supervising every hidden layer | adds opaque cost and muddies interpretation |
| folding step gating into the loss kernel | mixes policy with math |
| adding a predictor head “for research completeness” | breaks the cheap, narrow contract that makes STP practical |

The stack also has to stay honest about architecture differences. In hybrid layouts with attention, Mamba, expert, and recurrent blocks, not every layer family necessarily wants the same geometry prior. The local notation used elsewhere in the stack is helpful here: `A` means attention, `M` means Mamba, `E` means expert or MoE, and `R` means recurrent. A pattern like `NAM56R` or `AEMEAEMEAEMR` is not just branding. It is a reminder that layer selection is architecture-aware policy.

That does not mean STP must become block-type-specific on day one. It means the config should leave room for that discussion instead of pretending one global default is always correct.

## What should be ablated before calling it “done”

The current code is ready for disciplined ablations because the knobs are already separated.

| Ablation | Files that support it | What it answers |
| --- | --- | --- |
| `n_spans=1` vs `n_spans>1` | public STP loss sample, sanitized public test excerpts | how noisy the estimator is at fixed layer choice |
| final layer vs explicit list | public STP loss sample, runtime config surfaces | whether late semantic states or intermediate states benefit more |
| early start vs delayed start | public training runtime gating | whether STP helps only after a base representation stabilizes |
| dense-only vs hybrid patterns | runtime presets and model pattern naming | whether the objective behaves differently across `A/M/E/R` mixtures |

Two engineering facts should guide those ablations.

First, pipeline-parallel runtime notes already say auxiliary losses such as STP are dropped in that mode. So any headline about STP effectiveness must name the runtime context; otherwise the comparison can be false even if the config looks the same.

Second, the current tests are unit tests, not training-value proofs. They show that the objective is stable, differentiable, and shape-safe. They do not yet prove that a particular layer subset or warmup schedule improves downstream quality. That distinction should be preserved in the article and in any future preset docs.

## The right conclusion

The strongest thing about the current geodesic-loss design is not novelty. It is restraint. The prototype did not try to solve trajectory learning with a large auxiliary subsystem. It chose a local curvature penalty that is mathematically legible, cheap to compute, and compatible with the backend constraints already enforced elsewhere in the stack.

That is exactly why it is a plausible feature for MegaCpp. The landing should keep the kernel narrow, keep the layer list explicit, keep start-step policy in the runtime, and evaluate the feature as a configurable regularizer rather than a grand theory of representation geometry. If a future preset promotes it, the case should be made with architecture-specific receipts, not with generic claims.

## References

- sanitized public STP loss sample
- sanitized public test excerpts
- public runtime integration notes
- public TPU bringup notes
- sanitized public receipt excerpt
