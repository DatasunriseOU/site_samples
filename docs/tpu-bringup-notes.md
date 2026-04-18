# TPU Bringup Notes

This note summarizes a public-safe TPU bringup workflow for training
experiments.

Key themes:
- isolate runtime issues from model issues
- keep flag changes small and auditable
- separate host bringup, XLA graph stability, and data pipeline checks

Observed pressure points:
- graph recompilation after seemingly small shape drift
- runtime disagreement between JAX-side utilities and PyTorch/XLA execution
- memory spikes from long-lived activations rather than parameter count alone
- debugging noise when multiple precision or sharding changes land together

Working habits that helped:
- change one runtime dimension at a time
- keep a small deterministic smoke lane
- log stable labels for compile, execute, and input-pipeline phases
- write down the current pass/fail frontier after every useful repro
