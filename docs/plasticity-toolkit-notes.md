# Plasticity Toolkit Notes

This note summarizes the public FIRE, DASH, and ReDo integration story
without local paths or machine-specific details.

## What stays separate

FIRE, DASH, and ReDo are treated as separate tools because they operate on
different schedules and different failure modes.

- FIRE is a phase-boundary intervention for selected 2D weight matrices.
- DASH is a periodic directional-shrinkage rule applied during training.
- ReDo is a dormant-neuron recycle pass driven by activity tracking.

The useful engineering result is the separation itself: one toolkit, but three
different cadences and three different kinds of evidence.

## What the public sample supports

The public control-surface excerpt shows the combined scheduler shape:

- `apply_fire(...)` is guarded by an explicit step schedule.
- `reset_optimizer_states_for_fired_params(...)` runs immediately after FIRE.
- `dash_step(...)` is interval-driven.
- `redo_dormant_neurons(...)` is also interval-driven, with its own threshold.

That is enough to support the public story that FIRE is not “always on,” and
that DASH and ReDo are not the same maintenance move.

## Safety posture

The public story should keep three safety constraints visible.

1. Mutations happen on the local parameter view that the optimizer actually
   owns, not on an imagined dense global tensor.
2. Optimizer state must be reset for parameters rewritten by FIRE, otherwise
   the next optimizer step can partially undo the intervention.
3. TPU/XLA lanes should be described conservatively. Parameter surgery is most
   defensible on explicit local-tensor paths, not on compiler-fragile compiled
   lanes.

## Activation-family implications

ReDo is not equally useful under every activation family. The safe public
framing is simple: dormant-neuron recycling matters most on activation paths
that can actually accumulate inactive units over time. That is why ReDo should
be discussed as a selective maintenance tool, not as a universal default.

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/research/fire/fire-plasticity-toolkit__fire_dash_redo_surface__v1.py
- https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/docs/research/fire/fire-dash-redo-in-practice__toolkit_notes__v1.md
- https://arxiv.org/abs/2602.08040
- https://arxiv.org/abs/2410.23495
- https://arxiv.org/abs/2302.12902
