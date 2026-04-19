---
title: "ZeRO-3-shaped sharding on the XLA backend: what transfers from FSDP2 and what does not"
description: "How to think about TPU XLA sharding honestly: keep the ZeRO-3 memory goal, drop the assumption that TPU uses the same eager FSDP2 wrapper model as CUDA."
date: "2026-04-18"
tags: ["tpu", "xla", "spmd", "fsdp2", "zero-3"]
---

Teams often use "FSDP2 on TPU" as shorthand for a memory goal rather than a literal implementation. That shorthand is easy to misuse. On CUDA, FSDP2 is an eager wrapper and hook-based abstraction. On TPU XLA, the practical analogue is usually SPMD parameter sharding with ZeRO-3-like memory behavior, not the same wrapper mechanism.

## The important distinction

On CUDA, `fully_shard` rewrites module structure and installs runtime hooks for all-gather and reduce-scatter. On TPU XLA, sharding is generally expressed through SPMD annotations and compiler-owned collective placement. The memory objective may be similar, but the mechanism is different.

That is the right public framing:

- CUDA path: eager FSDP or FSDP2-style wrapper semantics
- TPU path: XLA SPMD sharding that aims for similar memory savings

Treating them as identical leads to bad debugging assumptions.

## What transfers cleanly

Some ideas do transfer across backends:

- classify which parameters should be sharded versus replicated
- keep the sharding policy stable across steps
- gate launches on whether the intended shard plan is actually valid
- separate memory goals from wrapper-specific implementation details

These are policy ideas, not proof that the same API surface exists on both backends.

## What does not transfer cleanly

Several familiar CUDA knobs do not map directly to TPU XLA:

- eager hook timing
- `reshard_after_forward`
- prefetch knobs tied to Python wrapper execution
- assumptions about local wrapper state being visible at every block boundary

On TPU XLA, collective placement and resharding behavior are compiler-shaped. The relevant debugging surfaces are graph stability, mesh construction, annotation correctness, and recompilation risk.

## Why this matters operationally

If a team says "FSDP2 on TPU" but is really using XLA SPMD sharding, then launch, profiling, and failure interpretation should follow the TPU model:

- confirm mesh and sharding annotations early
- keep the shard contract stable across steps
- treat recompilation and memory-space assignment as first-class risks
- avoid copying CUDA-only tuning vocabulary into TPU launch documentation

That keeps the operational story honest. It also avoids implying official parity where the underlying implementation model is different.

## A safer naming convention

For public docs, a safer pattern is:

- "FSDP2 on CUDA"
- "ZeRO-3-shaped sharding on TPU XLA"

That keeps the memory intent visible without claiming identical runtime machinery.

## References

- https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- https://docs.pytorch.org/xla/master/spmd.html
- https://research.google/blog/general-and-scalable-parallelization-for-neural-networks/
- https://openxla.org/shardy/getting_started_jax
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_flag_profile.py
