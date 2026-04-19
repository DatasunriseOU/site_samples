# FIRE And Plasticity Examples

This directory collects public-safe MegaCpp POC examples for FIRE, DASH, and
ReDo style plasticity tools.

What is here:
- `fire_sample.py`: target selection for topology-aware FIRE resets
- `fire_optimizer_reset_sample.py`: optimizer-state reset surface used together
  with FIRE
- `fire_context_extension_filter_sample.py`: the narrow context-extension filter
  for Q/K-style reset targets
- `dash_sample.py`: drift-aware selective reset scheduling
- `redo_sample.py`: dormant-unit recovery path
- `plasticity_phase_schedule_sample.py`: combined FIRE + DASH + ReDo cadence
  planning

What problem these samples solve:
- they let training recover from stale or over-specialized weights without
  throwing away the whole run
- they show which parameter surfaces are safe to touch during reset phases

Where this fits in the model:
- these helpers sit beside the optimizer and training schedule
- they are used during phase changes, context-extension ramps, and recovery
  experiments

In simple words:
- FIRE is the one-time phase surgery tool
- DASH is the periodic shrink pass for overconfident rows
- ReDo is the periodic dormant-neuron recycle pass
- the phase scheduler keeps those tools from firing on the same cadence
