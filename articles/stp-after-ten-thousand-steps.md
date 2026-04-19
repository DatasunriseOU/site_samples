---
title: "STP after ten thousand steps: what changed, what signal we watched, and what stayed"
description: "What the STP-style auxiliary loss can change once a run is past the early warmup window: the hidden-state straightness signal we monitor, why the main loss still dominates, and which parts of the baseline training recipe remain intentionally unchanged."
date: "2026-04-18"
tags: ["training", "stp", "trajectory-straightness", "auxiliary-loss", "representation-learning"]
---

# STP after ten thousand steps: what changed, what signal we watched, and what stayed

In this training policy, STP is evaluated after the early warmup window rather than at startup. Around the 10K-step gate, the useful interpretation is not that a new training regime has taken over, but that a bounded auxiliary geometry term may now be shaping hidden trajectories while the main next-token loss remains dominant. The activation-boundary note is in [the STP gate note](../excerpts/docs/research/stp/stp-after-ten-thousand-steps__activation_gate_note__v1.md).

Auxiliary objectives are easy to oversell because their motivation often sounds cleaner than their deployment reality. STP, in this naming scheme, refers to a trajectory-straightness auxiliary loss. The idea is straightforward: hidden states along a sequence should evolve along locally coherent paths rather than wobbling arbitrarily from token to token. That is a meaningful idea for code models, which are constantly asked to preserve structural intent across long spans while still responding to local syntax and semantics.

But “meaningful idea” is not the same thing as “production recipe.” The honest status is narrower. The feature exists, has tests, has runtime wiring surfaces, and is worth studying after the run is no longer dominated by cold-start effects. That is why the ten-thousand-step mark matters. It is late enough that representation geometry starts to become interpretable and early enough that the team can still compare shaped versus unshaped trajectories without waiting for a full production-scale campaign.

The threshold should therefore be read as an evaluation boundary, not as a mystical number. It marks the point where the team can stop blaming every movement on warmup turbulence and start asking whether the auxiliary term is doing anything identifiable.

## What STP is actually trying to shape

The implementation view matters more than the slogan. STP is not trying to replace next-token prediction. It is trying to add a geometric preference over hidden-state motion. The loss samples positions in the sequence, forms local directional relationships, and penalizes curvature by comparing the direction into a point and the direction out of it.

That gives you a bounded auxiliary signal rather than an unconstrained second objective. The value of that boundedness is practical: the team can reason about STP as a shaping term instead of a rival training program.

That boundedness also keeps collaboration saner. Infra engineers can ask whether runtime cost changed. Model engineers can ask whether geometry changed. Neither side has to pretend STP replaced the main optimization objective.

| Training phase | What STP might legitimately do | What it should not be expected to do |
| --- | --- | --- |
| Very early run | Add weak, mostly noisy geometric pressure | Produce stable conclusions about representation quality |
| Around the configured 10K-step gate | Start affecting sampled hidden-state straightness in a measurable way | Rewrite the base loss story |
| Mature run | Continue acting as a mild representation regularizer | Deliver quality wins without careful scheduling or ablation |

That table is the practical reason the article is about ten thousand steps rather than step zero. Before then, optimization transients dominate. After then, you can at least ask whether the hidden-state geometry is moving in the intended direction.

## Why a code model would care at all

For a code model, hidden-state geometry is not an abstract aesthetic issue. Code is full of long-range constraints: identifiers introduced hundreds of tokens earlier, control-flow branches that must reconverge semantically, type relationships that stay live across large contexts, and multi-line edits that change local syntax without changing the larger intention of a function or file.

A shaping loss that encourages locally straighter hidden-state trajectories is attractive because it could, in principle, reduce unnecessary representational zig-zagging while preserving the base language-model objective. That is why STP belongs in the training stack instead of being dismissed as decorative regularization.

This is even more relevant on hybrid architectures. A lane that mixes `A`, `M`, `E`, and `R` regions is already asking its hidden states to carry several kinds of structure. If STP helps, the team will likely want to know whether it helps uniformly or whether it mostly affects one family of blocks.

At the same time, this is exactly the sort of feature that can become self-deception if you do not bind it to runtime discipline. The project already has enough evidence surfaces to avoid that trap: launcher arguments, throughput tracking, receipt machinery, and a culture of separating “exists in code” from “deserves a maintained preset.” STP should be judged inside that discipline, not outside it.

## What should stay unchanged when STP turns on

One of the best rules for auxiliary losses is to preserve as much of the baseline recipe as possible. The point is to isolate what the new term is actually doing. Once you change ten things at once, any geometry story becomes suspect.

For STP, the baseline invariants should be boring:

- keep the main next-token loss path unchanged,
- keep the optimizer and schedule unchanged unless STP itself requires a narrow gating change,
- keep throughput and memory instrumentation on,
- delay activation until the run has reached a more interpretable regime,
- preserve the exact model and block naming in the receipts.

The last item matters more than it sounds. A run on a `NAM52` dense lane and a run on a `NAM56R` hybrid lane are not equivalent contexts for STP. Neither are runs on different block patterns such as `AEMEAEMEAEMR`. If hidden-state geometry changes, the architecture mix behind that change has to remain visible.

This is the same naming discipline that helps runtime bring-up receipts, but for a different reason. There it protects reproducibility. Here it protects interpretability.

```yaml
stp:
  enabled: true
  start_step: 10000
  lambda: 0.05
  n_spans: 4
  layers: "final"
observability:
  report: true
  temporal_perf: true
  receipts: true
```

That configuration block is illustrative, not copied from a checked-in file. It shows the shape of a responsible STP experiment: delayed activation, explicit weight, explicit span budget, explicit layer scope, and explicit observability.

## What signal we should actually watch

The wrong signal to obsess over is the STP scalar by itself. Auxiliary losses can look clean numerically while doing nothing useful or while quietly taxing throughput and stability. The right signals come in a bundle.

| Signal | Why it matters |
| --- | --- |
| Base next-token loss | If this regresses badly, STP is not paying its rent |
| Hidden-state straightness metric | This is the direct target of the auxiliary idea |
| Throughput / tok-sec / goodput | A geometry win that destroys training efficiency is not a free improvement |
| Peak memory | Auxiliary sampling and bookkeeping can add real cost |
| Receipt stability | If the experiment is not recorded structurally, the result will not survive handoff |

MegaCpp already has enough observability surfaces to support that bundle. Training telemetry can distinguish useful optimization time from compile or idle time, track throughput and memory across steps, and turn run artifacts into stable summaries. That is exactly the environment where STP can be evaluated honestly.

It is also the environment that prevents cherry-picking. A nicer auxiliary curve is not persuasive if useful training throughput drops sharply. Likewise, a tiny runtime tax is not persuasive if the hidden-state geometry does not move in a meaningful way.

The ten-thousand-step framing also helps here. If STP is delayed until after the coldest part of the run, then any change in hidden-state geometry is easier to attribute, and any runtime tax becomes easier to see against a more stable baseline.

## How STP should land in MegaCpp, if it does

MegaCpp should not inherit STP as a baseline feature just because the idea is attractive. It should inherit the experimental discipline around it.

That means three things.

First, STP should remain a weighted auxiliary term on top of the main loss, not a replacement objective and not a justification for changing the whole training recipe.

Second, activation should be schedule-aware. The right contract is “this term becomes active after a chosen step” rather than “this term is always on from token one.” The ten-thousand-step framing is the right kind of caution.

Third, MegaCpp should require the same receipt discipline as any other nontrivial feature. If an STP run cannot produce a stable report with exact model naming, throughput context, and a clear before/after comparison, it is not ready to influence production defaults.

That also implies a modest rollout strategy. The first valuable STP result is not “turn it on everywhere.” It is “we now know how it behaves on one exact, well-instrumented lane.”

This matters especially on hybrid architectures. A model with `A`, `M`, `E`, and `R` regions is already mixing different representational dynamics. If STP helps, the team will want to know whether it helps uniformly or whether it is mostly cleaning up a subset of layers. That is another reason to keep layer scope explicit and receipts exact.

## The biggest risk: confusing neat theory with operational value

The most likely way to get STP wrong is not an implementation bug. It is a framing bug. The feature is mathematically neat enough that people may start telling a bigger story than the evidence supports. The correct story is smaller and more useful.

Training history is full of features that sounded transformative until they met step time, memory budgets, and handoff requirements. STP should be protected from that fate by being described precisely and evaluated conservatively.

STP is a bounded geometry regularizer that may make hidden-state trajectories locally straighter once the run is warm enough for that question to mean something. It should be judged by whether that effect appears without unacceptable tax on throughput, memory, and base optimization behavior.

That smaller story is not disappointing. It is the only one likely to survive contact with real runs.

## What we would keep even if STP stays experimental

Even if STP never becomes a default feature, the work around it still pays off. It pushes the stack toward better delayed-activation controls, better experiment receipts, clearer thinking about hidden-state geometry, and more honest feature evaluation. Those are good outcomes independent of the final verdict.

It also sharpens how the team should talk about auxiliary objectives generally. The real question is not “does this idea sound principled?” It is “what moved after the run was stable enough for the movement to mean something?”

That is another reason the ten-thousand-step lens is useful. It keeps the discussion attached to runtime behavior instead of drifting into aesthetic arguments about auxiliary losses.

## Code and notes

- [STP activation gate note](../excerpts/docs/research/stp/stp-after-ten-thousand-steps__activation_gate_note__v1.md)
- [STP loss surface sample](../excerpts/code/research/stp/stp-geodesic-regularizer__stp_loss_surface__v1.py)

## Further reading

- [MegaCpp STP activation note](../excerpts/docs/research/stp/stp-after-ten-thousand-steps__activation_gate_note__v1.md)
- [MegaCpp STP loss sample](../excerpts/code/research/stp/stp-geodesic-regularizer__stp_loss_surface__v1.py)
