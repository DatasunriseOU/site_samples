# FIRE Index

This folder covers the MegaCpp POC plasticity toolkit.

- `fire_sample.py`: choose the 2D weights that should be reset.
- `fire_optimizer_reset_sample.py`: reset matching optimizer slots.
- `dash_sample.py`: score which parts of the model should be refreshed first.
- `redo_sample.py`: revive dormant units instead of leaving them permanently
  inactive.

In simple terms: FIRE is the direct reset tool, DASH helps decide when and
where to use it, and ReDo is the narrow dormant-unit repair path.
