---
title: "Attention sinks and telemetry on TPU: measure without turning observability into the bug"
description: "Why TPU telemetry has to be gated carefully: scalar reads can become host-device syncs, so sink and outlier tracking must be designed as explicit low-cadence instrumentation."
date: "2026-04-18"
tags: ["tpu", "telemetry", "xla", "attention", "observability"]
---

Telemetry is useful only if it does not distort the run it is trying to measure. On TPU XLA, that is a real risk because scalar extraction inside the hot path can trigger host-device synchronization.

## Why TPU telemetry needs stricter discipline

A naive attention-sink tracker can look harmless in Python while quietly introducing step-time regressions. On XLA, calls such as `.item()` are often exactly the kind of scalar boundary that should not live inside a high-frequency compiled path.

That leads to a simple operational rule: high-detail telemetry should be gated, not left permanently on.

## What the instrumentation should do

The useful split is:

- a sink-oriented stream for attention concentration
- an activation-oriented stream for outliers or spikes

The important part is not one exact implementation. It is the execution discipline:

- no-op on non-logging steps
- bounded work on logging steps
- explicit operator-visible cadence

## What a good TPU telemetry surface looks like

For TPU training, the cleanest telemetry API is usually:

- attach once
- enable or disable per logging cadence
- expose structured summaries rather than ad hoc scalar prints

That keeps observability compatible with compiled execution.

## What should be preserved

The core idea worth preserving is that telemetry is part of the runtime contract. If instrumentation can materially change step time, then cadence, gating, and summary shape should all be treated as first-class policy rather than debugging leftovers.

## References

- https://docs.pytorch.org/xla/master/perf/recompilation.html
- https://docs.pytorch.org/xla/master/runtime.html
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_flag_profile.py
- https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/research/fire/fire-plasticity-toolkit__fire_dash_redo_surface__v1.py
