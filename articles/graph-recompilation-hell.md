---
title: "Graph recompilation hell: shape drift, graph contracts, and why TPU runs slow down without crashing"
description: "A walkthrough of the most common TPU recompilation failure mode: changing shapes, unstable graph contracts, and weak runtime discipline."
date: "2026-04-18"
tags: ["xla", "tpu", "recompilation", "graph", "performance"]
---

The most expensive TPU failures are often not crashes. They are recompiles: the run stays alive, step time stretches, the compile cache stops helping, and the team spends hours looking at the wrong layer. In practice this is usually a graph-contract problem, not a random TPU failure.

## What public XLA guidance already says

PyTorch/XLA's public recompilation notes are direct: XLA prefers static shapes, and changing shapes can trigger recompilation. The bounded-dynamic-shape docs soften that story, but they do not erase it. Bounded dynamic shape reduces some classes of recompilation. It does not make graph drift disappear.

That matters because recompilation is often described as if it were random. It usually is not. It is a symptom that the runtime is seeing a materially different graph contract.

## The most common causes

| Cause                        | Why it recompiles                                      |
| ---------------------------- | ------------------------------------------------------ |
| input shape drift            | XLA sees a different graph signature                   |
| data-dependent graph changes | the traced program no longer matches the previous step |
| hidden startup inconsistency | runtime policy changed between launches                |
| weak batching discipline     | supposedly identical steps do not really match         |

## Why dynamic shape is not a magic fix

Bounded dynamic shape is useful on TPU because it can absorb some variation while keeping memory allocation compatible with accelerator constraints. But the public docs are clear: it reduces some recompiles, not all of them.

That is why the practical debugging rule is to change one runtime dimension at a time and keep a small deterministic smoke lane. If several things move together, recompilation stops being diagnosable.

## Practical rule

If a TPU run slows down without crashing, ask three questions before touching model math:

1. Did input shapes change?
2. Did the runtime profile or SPMD policy change?
3. Did a previously hidden graph branch become active?

Most of the time, one of those answers is yes.

## References

- https://pytorch.org/xla/master/notes/source_of_recompilation.html
- https://docs.pytorch.org/xla/master/perf/recompilation.html
- https://docs.pytorch.org/xla/master/learn/dynamic_shape.html
