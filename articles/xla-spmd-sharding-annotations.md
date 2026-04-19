---
title: "XLA SPMD sharding annotations we actually rely on"
description: "Why explicit `mark_sharding` annotations matter on TPU XLA, what should be pinned explicitly, and why propagation is not a substitute for a stable sharding contract."
date: "2026-04-18"
tags: ["xla", "spmd", "tpu", "sharding"]
---

This post is about the annotation surface itself: where explicit `mark_sharding` calls matter, where replication should be stated out loud, and why leaving parameter placement to inference is often the wrong tradeoff on TPU XLA.

## Why propagation is not enough

SPMD propagation is useful, but it is not a replacement for an explicit parameter contract. If a parameter is left unannotated, XLA is free to infer a placement from surrounding graph structure. Sometimes that is fine. Sometimes it is wrong in a way that hurts correctness or compile stability rather than crashing immediately.

That is why the conservative rule is simple: explicitly annotate parameters you own, including the ones you intend to replicate.

## What should be explicit

Three classes benefit most from explicit annotations:

- parameters that should be sharded along a known mesh axis
- parameters that must remain replicated even if their shape happens to divide cleanly
- boundary activations whose layout is part of the intended compile contract

The important idea is not one exact project-specific whitelist. It is the operational discipline: if a tensor layout matters, pin it.

## Why replication should also be stated explicitly

Small tensors are often the easiest place to make a bad assumption. A projection or auxiliary parameter may coincidentally match a mesh axis and look shardable even when the intended semantics are replicated. In that situation, explicit replication is safer than letting inference guess.

```python
import torch_xla.distributed.spmd as xs

def replicated(ndim: int):
    return tuple(None for _ in range(ndim))

for name, param in model.named_parameters():
    if name in REPLICATED_NAMES:
        xs.mark_sharding(param, mesh, replicated(param.ndim))
```

## What a good audit looks like

If sharding is part of the startup contract, it should be auditable. A useful audit checks that:

- every intended parameter annotation was applied
- replication is explicit where it is supposed to be explicit
- mesh shape and axis names match the launch configuration

This matters because a sharding mistake can show up as compile instability or subtle training drift rather than as a loud startup failure.

## A safer public claim

The useful public claim is not that one exact sharding table is universal. The useful claim is narrower:

- on TPU XLA, parameter placement should be explicit where semantics matter
- replication is a real placement choice and should be stated explicitly
- propagation is helpful for intermediates, but risky as the sole source of truth for owned parameters

## References

- https://docs.pytorch.org/xla/master/spmd.html
- https://docs.pytorch.org/xla/master/perf/recompilation.html
- https://research.google/blog/general-and-scalable-parallelization-for-neural-networks/
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_flag_profile.py
