---
title: "Mamba-3 fused trapezoidal scan on TPU v6e"
description: "How we took the Mamba-3 trapezoidal SSM update from a CUDA Triton kernel to a Pallas/XLA-friendly scan on TPU v6e, and what survived the deployment port."
date: "2026-04-18"
tags: ["mamba3", "tpu", "v6e", "pallas", "xla", "ssm"]
---

MegaCpp uses Mamba-style blocks because they let a small model spend part of
its budget on sequence mixing that is not just another attention layer. On TPU,
that only works if the state-space update stays inside a compile-stable path.

This article is about that TPU constraint, not about claiming a universal
"best" kernel.

## The engineering problem

The Mamba-3 paper is public. So are PyTorch/XLA runtime docs and JAX's TPU
Pallas docs. What those sources collectively support is a narrow claim:

- Mamba-3 is a real model family with a state-space core
- PyTorch/XLA rewards static-shape-friendly compiled execution
- Pallas can target TPU, but it is documented as experimental

MegaCpp's TPU porting story sits inside that triangle. The real work is not
proving that one exact kernel is universally optimal. The real work is keeping
the scan, the boundary metadata, and the surrounding elementwise work in a path
that does not recompile or materialize avoidable temporary tensors.

## Why the update shape matters

Once a state-space block depends on adjacent timestep information, a naive
implementation quickly becomes launch-heavy. The high-level problem is easy to
state:

- the scan itself is not the only cost
- prologue and epilogue work can dominate if they are split into many tiny ops
- TPU performance suffers if the path becomes shape-dynamic or mask-heavy

MegaCpp's TPU discipline is therefore:

1. keep the scan body static-shape-friendly
2. materialize boundary metadata explicitly
3. let XLA fuse surrounding elementwise work by default
4. only introduce a custom TPU kernel when it removes a real hot-path cost

## Why boundary metadata belongs inside the compiled path

Packed training data made this non-optional. If document boundaries are part of
the training contract, the state-space path cannot treat them as an afterthought.
MegaCpp therefore keeps sequence or segment identifiers explicit instead of
relying on hidden implicit state.

That choice matters for two reasons:

- it makes the compile-time shape story easier to reason about
- it keeps document-boundary semantics aligned with the rest of the long-context pipeline

Publicly, the useful claim is not "our exact implementation uses variable X."
The useful claim is that sequence-boundary metadata must travel with the
compiled path if you want long-context training to stay correct.

## Where XLA is enough and where Pallas is considered

The default MegaCpp TPU stance is conservative: plain XLA fusion around the
compiled scan is the baseline. Pallas is considered only when a custom tiled
kernel clearly removes extra passes or makes boundary handling materially
cleaner.

That is a narrower and more defensible policy than saying "we rewrote the whole
thing in Pallas." In practice, many TPU performance problems are better solved
by keeping the path static and letting XLA do its job than by taking ownership
of another kernel surface.

## The rules that survived the port

The TPU-friendly rules are:

- keep chunking or scan-shape choices explicit and stable
- keep boundary identifiers materialized instead of implicit
- avoid dynamic per-step kernel configuration
- prefer one compiled path over a Python bridge plus extra runtime glue
- treat custom TPU kernels as opt-in, not as the default answer

These rules follow directly from the official TPU
runtime and Pallas documentation, even though the exact MegaCpp code path is
project-specific.

## What we avoid claiming

MegaCpp does **not** use this article to claim:

- that TPU is the canonical home of Mamba-3
- that Pallas is the settled production path for every TPU kernel
- that one exact fused update is the only correct implementation

Those claims would go beyond the public sources. The safer statement is that
MegaCpp adapted a Mamba-style scan to TPU by favoring compile stability,
explicit boundary metadata, and narrow use of custom kernels.

## References

- [Mamba-3 porting note](https://github.com/DatasunriseOU/site_samples/blob/main/docs/mamba3-trapezoid-porting.md)
- [TPU backend ownership note](https://github.com/DatasunriseOU/site_samples/blob/main/docs/tpu-backend-ownership.md)
- [Mamba-3 paper](https://arxiv.org/abs/2603.15569)
- [PyTorch/XLA runtime docs](https://docs.pytorch.org/xla/master/runtime.html)
- [JAX Pallas on TPU](https://docs.jax.dev/en/latest/pallas/tpu/)
- [JAX Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
