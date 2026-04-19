# FIRE Index

This folder covers the MegaCpp POC plasticity toolkit.

- `fire_sample.py`: choose the 2D weights that should be reset.
- `fire_optimizer_reset_sample.py`: reset matching optimizer slots.
- `fire_context_extension_filter_sample.py`: narrow the phase transition to the
  Q/K-style attention targets that matter for context extension.
- `dash_sample.py`: score which parts of the model should be refreshed first.
- `redo_sample.py`: revive dormant units instead of leaving them permanently
  inactive.
- `plasticity_phase_schedule_sample.py`: keep the one-shot and periodic tools on
  separate schedules.

In simple terms: FIRE is the direct reset tool, DASH helps decide when and
where to use it, ReDo is the narrow dormant-unit repair path, and the scheduler
keeps the three tools coordinated.
