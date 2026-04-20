---
title: "XLA-safe AdamW and TPU runtime flags on v6e"
description: "How to keep optimizer math graph-friendly on TPU, treat runtime flags as explicit launch policy, and recalibrate after stack changes."
date: "2026-04-18"
tags: ["tpu", "v6e", "xla", "adamw", "pjrt", "calibration"]
---

On accelerator backends, the AdamW step often sits near the boundary between traced tensor math and Python control flow. On Cloud TPU v6e, that boundary is one of the first places where scalar extraction, host-device sync, or shape drift can trigger recompilation. The practical fix is simple: keep scalar handling graph-friendly, treat TPU runtime flags as explicit launch policy, and recalibrate after stack changes.

## Why the optimizer step is the canary

The optimizer step sits at an awkward boundary between Python control flow and tensor math. On TPU that boundary matters because scalar extraction and changing-shape control flow are both compile-sensitive in XLA.

The rule is straightforward: values that change every step should remain visible to the traced program as tensors, not escape to Python scalars inside the hot path.

```python
# Stylized TPU-safe sketch.
def adamw_step_xla(p, g, exp_avg, exp_avg_sq, scalars):
    p.mul_(scalars["one_minus_lr_wd"])
    exp_avg.lerp_(g, scalars["one_minus_b1"])
    exp_avg_sq.mul_(scalars["b2"]).addcmul_(g, g, value=scalars["one_minus_b2"])
```

The exact implementation can vary. The important point is that the graph sees tensors and stable shapes rather than a stream of fresh Python values.

## Runtime flags should be policy, not shell state

The public XLA flag sample is intentionally small, but it captures the main point: flag changes should be deliberate, reviewable, and grouped by purpose rather than scattered across shell wrappers.

| Group                    | Representative policy     | Why it matters                                 |
| ------------------------ | ------------------------- | ---------------------------------------------- |
| SPMD enablement          | explicit startup flagging | makes mesh assumptions visible                 |
| compile cache policy     | explicit cache mode       | separates cold-start from steady-state effects |
| shape guard policy       | strict input contract     | reduces accidental recompile drift             |
| launch profile selection | named runtime profile     | keeps runs comparable                          |

The takeaway is not that one magic flag profile solves TPU performance. It is that runtime policy should be explicit and narrow enough that when performance changes, you can tell whether the cause was the graph, the inputs, or the runtime profile.

That is also why the launcher should own the runtime profile before importing the TPU runtime. Import order is part of correctness here, not only style.

## Calibration matters after stack changes

A small startup calibration is cheaper than repeating a long failing launch. What matters is recording predicted versus observed memory behavior and feeding that back into the next run.

That loop is what makes runtime-policy changes survivable across stack upgrades. When the TPU runtime changes behavior, the calibration record should catch the mismatch before a large run turns into an avoidable OOM or recompilation storm.

## Takeaway

The TPU optimizer story is not really about one exotic optimizer trick. It is about respecting the graph contract, keeping runtime policy explicit, and recalibrating when the stack changes.

## References

- https://docs.pytorch.org/xla/master/runtime.html
- https://docs.pytorch.org/xla/master/perf/recompilation.html
- https://openxla.org/xla/oom_debugging
- https://cloud.google.com/tpu/docs/v6e
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_safe_adamw.py
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_flag_profile.py
