---
title: "Pallas kernels on TPU v6e: what we ship and what we deleted"
description: "Where Pallas beats the XLA lowering on TPU v6e, where it loses, the debugging workflow that keeps us sane, and the kernel deltas we kept versus the ones we reverted."
date: "2026-04-18"
tags: ["pallas", "tpu", "v6e", "jax"]
---

Pallas is useful on TPU, but the public documentation is clear about one thing:
it is still an experimental kernel-writing surface. MegaCpp therefore treats it
as a narrow TPU-only tool, not as the default answer to every optimization
problem and not as a substitute for NVIDIA or NVFP4-specific kernel work.

## The rule that actually matters

The important question is not "can we write this in Pallas?" The important
question is "does writing this in Pallas buy us something XLA lowering does not
already give us?"

MegaCpp keeps a short decision rule:

- prefer XLA lowering when the default path is already good enough
- prefer Pallas when tile control, local-window structure, or segment-aware
  masking removes a real hot-path cost

That rule is stricter than it sounds. A custom kernel is not just another
function. It becomes part of the compile contract, the sharding story, and the
upgrade surface.

## Where Pallas earns its keep

Publicly defensible use cases are the ones where the custom kernel surface is
obviously doing something structural:

- keeping softcap or local-window logic inside the hot loop
- using segment ids instead of materializing a dense mask
- keeping block structure explicit rather than relying on a generic dense fallback
- holding tile sizes fixed so the step does not recompile

Those are the kinds of cases where a custom TPU kernel can earn its maintenance
budget.

## Where XLA is the right answer

MegaCpp leaves many paths alone:

- plain dense attention
- short-sequence cases where compile overhead dominates
- norms and similar reduction-heavy operations that XLA already fuses well
- dynamic-shape stories that would turn every step into a retrace or recompile

That is an important part of the public claim. A TPU stack becomes harder to
trust when every path is rewritten just because a lower-level tool exists.

## The workflow that survived

The debugging workflow is simple:

1. keep mask or layout logic reproducible on CPU first
2. compare custom-kernel outputs against a trusted reference path
3. record which backend actually executed
4. only promote the kernel if it wins clearly enough to justify its maintenance cost

This is why MegaCpp prefers explicit backend receipts. A TPU configuration
should never need log archaeology to answer the question "did the custom kernel
actually run?" The article should also make it obvious that this is a TPU and
JAX/Pallas lane, not an NVIDIA precision or CUDA-kernel lane.

## Why MegaCpp is conservative here

Pallas can be excellent for narrow cases, but TPU execution already has enough
moving parts: frontend behavior, PJRT, runtime versions, compile caches, and
sharding behavior. Every unnecessary custom kernel makes that matrix harder to
reason about.

MegaCpp therefore keeps a bias toward deletion:

- if a Pallas path is only tied with XLA, it should probably be removed
- if a Pallas path helps only at one very narrow shape, it should stay
  experimental until the use case is stable
- if a Pallas path requires dynamic per-step shape choices, it should probably
  stay out of the training hot loop

## The public claim

The useful public statement is:

- MegaCpp uses Pallas selectively on TPU
- it keeps XLA lowering as the default for many paths
- it promotes a custom TPU kernel only when the structural win is clear
- it treats backend receipts and correctness checks as part of the deployment contract

That is consistent with the official Pallas docs and avoids presenting an
experimental surface as if it were a settled platform guarantee.

## References

- [Pallas kernel selection note](https://github.com/DatasunriseOU/site_samples/blob/main/docs/pallas-kernel-selection.md)
- [TPU bringup notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/tpu-bringup-notes.md)
- [JAX Pallas on TPU](https://docs.jax.dev/en/latest/pallas/tpu/)
- [JAX Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [JAX Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html)
- [Shardy for JAX users](https://openxla.org/shardy/getting_started_jax)
