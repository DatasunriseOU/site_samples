---
title: "FIRE, DASH, ReDo in practice: cadences, shard safety, and when we turn them off"
description: "How the prototype’s plasticity stack actually works in code: one-shot FIRE resets, periodic DASH and ReDo passes, DTensor-safe parameter surgery, and the training lanes where the whole toolkit is intentionally disabled."
date: "2026-04-18"
tags: ["training", "plasticity", "fire", "dash", "redo", "fsdp2", "dtensor"]
---

TL;DR: the plasticity stack is useful because it is operationally narrow. `FIRE` is a phase-boundary reset for selected 2D weights, `DASH` is a low-frequency row shrink pass guided by weight-gradient cosine, and `ReDo` is a dormant-neuron recycle pass driven by activation statistics. The implementation in the public FIRE module sample is more interesting than the paper names: it contains DTensor-local accessors, shard-bound math, optimizer-state reset hooks, and explicit XLA guards that tell you where parameter surgery is safe and where it is not.

## The real problem this stack is trying to solve

The prototype does not treat every long-run degradation as a learning-rate problem. The design notes in a model architecture design note make a broader point: once you mix attention, Mamba, expert layers, and recurrent-style blocks, the model can enter a regime where some subspaces need fresh geometry while the rest of the run still needs continuity. That is especially visible when a run crosses phase boundaries, changes context assumptions, or carries old projection geometry into a new curriculum stage.

This is why the plasticity tools are intentionally asymmetric. The code is not saying “reset everything more often.” It is saying that some weights should get a one-time geometric correction, some rows benefit from periodic shrinkage, and some dormant neurons should be recycled only after enough evidence accumulates. That split matters more in a sharded setup than it does in a single-GPU toy run. A mathematically clean intervention that breaks local shard ownership, leaves optimizer state stale, or mutates parameters in a compiled lane at the wrong time is not a stability tool. It is a new failure mode.

The implementation reflects that reality. the public FIRE module sample spends a lot of effort on helper surfaces such as `_local_tensor_if_dtensor`, `_is_dtensor_like`, `_dist_rank_world_size`, `_shard0_bounds`, and `_match_grad_to_local_shard`. Those are not helper clutter. They are the code’s admission that any serious plasticity pass must operate on the local shard seen by the running process, not on an imagined full tensor.

## What each intervention actually does

The three mechanisms live in one module because they share the same runtime constraints even though they do different jobs.

| Tool | Trigger style | Main target | Code-level behavior | Practical purpose |
| --- | --- | --- | --- | --- |
| `FIRE` | one-shot, usually at a phase boundary | selected 2D parameter matrices | Newton-Schulz orthogonalization in fp32 | reset geometry without repeatedly fighting the optimizer |
| `DASH` | periodic during training | rows of 2D weights | shrink rows whose weight-gradient cosine exceeds a threshold | soften overcommitted directions without full reinit |
| `ReDo` | periodic after enough activation history | dormant MLP neurons | resample incoming rows, perturb outgoing columns, zero biases | wake up neurons that effectively stopped participating |

`FIRE` is the most opinionated. The module-level docstring states the intended cadence directly: “Applied ONCE between training phases.” The transform itself is built around `newton_schulz`, which projects weight matrices toward an orthogonal form using fp32 math. The public path applies only to 2D parameters. That is a meaningful restriction. It means embeddings, scalar parameters, state tensors, and anything else that is not structurally a matrix stays outside the reset.

The targeting logic matters too. `get_fire_targets` provides selective modes rather than a blanket rewrite. In practice that lets the code support a broad aggressive pass or a narrower context-extension-oriented pass focused on attention-style projections. The design notes around long-context evolution in `03-model-architecture.md` are exactly the kind of regime where that narrower targeting makes sense: the model is not globally broken, but the geometry of certain projections can lag the new objective.

`DASH` is lighter weight and intentionally online. `dash_step` computes cosine similarity between each row of a weight and the corresponding gradient row. If the cosine is too high, the row is shrunk by a bounded factor. The mechanism is conservative because it is supposed to coexist with the optimizer rather than replace it. The row is not reinitialized. It is nudged away from a state where the current gradient keeps reinforcing the same already-dominant direction.

`ReDo` is more surgical. The `ReDoDiagnostics` path installs forward hooks on MLP-like modules and keeps an EMA of mean absolute activation. Once that evidence exists, `recycle_dormant_neurons` finds units whose activity falls below a threshold relative to the layer mean. The incoming row is reinitialized, the outgoing column gets only a smaller perturbation, and the matching bias is zeroed. That asymmetry is important. The implementation is trying to revive capacity without causing a giant downstream shock on the next step.

## Why the safety scaffolding matters more than the names

The easiest mistake in writing about this module is to focus on the algorithm labels and miss the distributed-systems logic. The difficult part is not inventing another reset rule. The difficult part is making the intervention compatible with DTensor, uneven shard sizes, and optimizer bookkeeping.

`_local_tensor_if_dtensor` is the first key surface. It checks for `_local_tensor`, falls back to `to_local()` when available, and otherwise returns the tensor as-is. That lets the same logic work across plain tensors, DTensor wrappers, and nightly-ish variants without pretending they all expose identical APIs.

`_shard0_bounds` is the second key piece. It computes rank-local row ranges for uneven `Shard(0)` partitioning by using the common base-plus-remainder split. That prevents rank-local FIRE or DASH passes from assuming even row ownership when the world size does not divide the row count cleanly.

`_match_grad_to_local_shard` closes the loop by ensuring that the gradient being used for the intervention matches the local parameter view. In other words, the module is not satisfied with “there is a gradient.” It wants the gradient in the same shard-local shape as the weight that is about to be rewritten.

The test file sanitized FIRE tests exists for precisely this reason. The tests are not just paper-equation checks. They encode contracts like “local tensor access works on DTensor-like wrappers,” “shard bound calculations are correct,” and “the intervention behaves correctly on the local shard rather than a fictional full parameter.” Those tests are what turn the feature from an experiment into an operational subsystem.

A second safety layer is device gating. `_is_xla_tensor` makes it explicit that XLA devices are not first-class targets for this surgery path. That aligns with the broader TPU/XLA bug record in a live bug audit report, where compiled XLA paths still have fragile grad materialization and cleanup behavior for conditional parameter sets. If the optimizer path already needs explicit static-grad handling to remain sound, adding mid-run parameter rewriting on top of it is the wrong place to be adventurous.

## When we run these tools, and when we do not

The code and docs together imply a practical cadence rather than a single universal default.

```yaml
plasticity:
  fire:
    enabled: true
    cadence: one-shot
    trigger: phase-boundary-or-explicit-intervention
    typical_targets: attention-style 2d weights or broader aggressive set
  dash:
    enabled: true
    cadence: low-frequency periodic
    preconditions:
      - stable gradient flow
      - local shard gradient available
  redo:
    enabled: true
    cadence: periodic after activation EMA is meaningful
    preconditions:
      - hook-based activation stats collected
      - mlp-like layer structure available
  skip_lanes:
    - xla-compiled training
    - unsafe compiled parameter-surgery paths
    - places where optimizer-state reset is not guaranteed
```

The one-shot nature of FIRE is not just a recommendation. It follows from the role the algorithm is playing. If you repeatedly re-orthogonalize matrices during the same optimization phase, you stop doing a reset and start fighting the optimizer. The module docstring says this plainly, and the rest of the targeting logic supports it.

DASH and ReDo are periodic because they depend on online evidence. DASH needs a meaningful gradient direction. ReDo needs an activation history with enough signal to distinguish true dormancy from a temporary lull. That means the toolkit is usually off during unstable bring-up and early-run chaos, even if the command-line surface technically allows it.

There is another practical skip rule: lanes with complicated compiler behavior. A compile-warmup regression note is a reminder that even without plasticity surgery, some `regional_compile + MoE` paths already need careful policy choices just to avoid bad warmup behavior. When a lane is still proving basic compile stability, adding in-place parameter interventions is the wrong trade.

## Where this fits in a NAM56R-style stack

A useful way to read the module is through the architecture vocabulary used elsewhere in the project: `A` for attention, `M` for Mamba, `E` for expert, `R` for recurrent. In a hybrid pattern such as `AEMEAEMEAEMR`, not every block has the same failure mode and not every block benefits from the same intervention.

That hybrid thinking shows up strongly in the restoration work on the Megatron-side port. the public NAM56R recipe sample and the public NAM56R Megatron recipe sample encode the pattern-aware model recipe. the public hybrid schedule sample goes further and makes the distinction runtime-visible: MoE-only layers are handled differently from Mamba or DSA-style opaque layers, and the schedule plan explicitly documents that NAM56R expert layers may have `IdentityOp` attention. That tells you something important for plasticity policy: a tool that is intuitively “for transformer attention blocks” is not automatically meaningful for every layer in a mixed pattern.

The result is a sensible operational split.

| Block family | Typical issue | Best-fitting intervention | Reason |
| --- | --- | --- | --- |
| `A` / attention-heavy 2D projections | stale geometry after phase or context shift | `FIRE` | projection reset is coherent and easy to target |
| `M` / Mamba-style state-space paths | drift can be structural, not just row dominance | usually avoid blanket plasticity | these paths are not just ordinary MLP matrices |
| `E` / expert MLP capacity | dormant sub-capacity or overcommitted rows | `DASH` and `ReDo` | row-level maintenance fits better than global reset |
| `R` / recurrent-style tail blocks | strongly regime-dependent | selective use only | recurrence-like blocks usually need extra caution |

That does not mean FIRE is impossible outside attention-style matrices. It means the codebase already gives you enough evidence to avoid the lazy “apply the same trick everywhere” instinct. Pattern-aware models need pattern-aware maintenance.

## What survived contact with reality

The strongest lesson from this module is not that all three interventions should always be on. It is almost the opposite. The implementation shows a mature bias toward narrow use.

The project kept the following ideas:

- only 2D matrices are eligible for FIRE;
- local DTensor shards are the unit of mutation;
- optimizer-state repair is part of the feature, not an optional afterthought;
- activation-driven recycling needs history, not a single batch snapshot;
- XLA paths should be treated conservatively when grad materialization is already fragile.

It implicitly rejected several weaker ideas:

- running FIRE continuously inside the same phase;
- pretending distributed tensors can be rewritten as if they were dense locals;
- treating every low-activation neuron as immediately safe to resample;
- using plasticity surgery in compiler-fragile bring-up lanes just because the feature exists.

That is what makes the module practically useful. It does not promise magic. It gives you a bounded set of interventions with enough runtime discipline to be believable.

## References

- the public FIRE module sample
- sanitized FIRE tests
- a model architecture design note
- a live bug audit report
- a compile warmup regression report
- the public NAM56R recipe sample
- the public NAM56R Megatron recipe sample
- the public hybrid schedule sample
