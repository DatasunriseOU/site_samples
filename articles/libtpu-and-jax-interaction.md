---
title: "libtpu and JAX interaction: shared runtime, separate ownership"
description: "How PyTorch/XLA, JAX, PJRT, and libtpu relate on TPU without collapsing distinct layers into one vague runtime claim."
date: "2026-04-18"
tags: ["libtpu", "jax", "torch-xla", "pjrt", "tpu"]
---

`libtpu` is part of the TPU runtime stack used by current TPU software. PyTorch/XLA uses PJRT as its runtime interface, and JAX also targets TPU through PJRT-based runtime paths. That means a mixed-tooling machine is not automatically broken, but it does mean version hygiene and ownership boundaries matter more than they do on a simpler CUDA-only box.

This is a TPU runtime article, not a CUDA or Blackwell article. It should not
be used to justify claims about NVFP4, Transformer Engine, or GPU kernel
behavior.

## A common wrong mental model

The wrong claim is: "JAX and PyTorch/XLA both use libtpu, therefore they inherently conflict." Public documentation does not support that. Both ecosystems expose TPU support through the same broad runtime family.

The more accurate framing is that both toolchains sit on a shared runtime substrate, so version drift, cache state, and mismatched assumptions can surface as hard-to-read failures.

## Where ownership really lives

| Layer              | Typical owner                                              |
| ------------------ | ---------------------------------------------------------- |
| application config | backend selection, fallback policy, logging                |
| framework frontend | PyTorch/XLA or JAX tracing and execution model             |
| PJRT runtime       | runtime interface between frontend and accelerator backend |
| TPU runtime stack  | device-specific execution and version compatibility        |

That table is the useful abstraction. Most TPU failures are not "the TPU does not exist" failures. They are ownership failures: the wrong backend was selected, a fallback hid the real problem, or a frontend/runtime mismatch was interpreted as a model bug.

## Why mixed-tooling environments still need discipline

A shared runtime does not mean a free-for-all. It means the operator should keep runtime selection explicit, record which frontend actually executed the run, and avoid silently inheriting shell state from unrelated tooling.

The goal is not to ban shared tooling. The goal is to make the active runtime contract legible.

## Practical takeaway

If a TPU lane is failing, debug in this order:

1. Which frontend actually owns execution for this run?
2. Which runtime contract is active: PJRT plus which TPU runtime stack?
3. Did a fallback or cache artifact hide the real boundary that failed?

That framing is more accurate than blaming JAX or PyTorch/XLA in the abstract.

## References

- https://docs.pytorch.org/xla/master/runtime.html
- https://docs.jax.dev/en/latest/installation.html
- https://docs.jax.dev/en/latest/pallas/quickstart.html
