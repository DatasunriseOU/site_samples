---
title: "External library glitches we fixed"
description: "A catalog of upstream bugs we hit while training our hybrid Mamba-3 plus DSA recipe, grouped by library: what broke, what we patched locally, and what we prepared upstream."
date: "2026-04-18"
author: "David Gornshtein"
tags: ["upstream", "debugging", "engineering"]
---

Running a hybrid Mamba-3 plus DSA recipe on a nightly PyTorch stack means every library in the fan-out is moving while we use it. This post is the public catalog: grouped by library, each entry names the symptom, our local workaround, and the upstream contribution we prepared or landed. The goal is not to expose every non-public breadcrumb. It is to show the engineering pattern clearly enough that another team could apply it.

## Why MegaCpp cares about this

A well-kept patch lane is one of the cheapest forms of institutional memory we have. When a training run goes NaN at iteration 3, we do not want to rediscover a bug we already fixed two weeks ago under a slightly different shape. The rule we enforce is simple: every local patch must point either to an upstream issue or PR, or to a public-facing draft that is ready to become one. Every retirement of a local patch must be explicit.

## What we built in practice

### PyTorch and Dynamo

We currently carry no source patches against `torch` itself. What we do carry is a narrow set of Dynamo configuration choices wired in early, together with explicit `torch._dynamo.disable` boundaries around kernels and routing paths that are not yet good compile citizens.

The load-bearing workaround here is configuration, not source divergence. One real torch-side glitch we still account for is an older `reduce_scatter_tensor` regression that showed up in Megatron tests on earlier versions. The fix there is a version floor, not a patch.

### `torch_xla` and `libtpu`

The TPU lane has its own class of regressions and does not overlap much with the GPU incidents in this catalog. The shared discipline is provenance: every TPU run records the exact `torch`, `torch_xla`, and `libtpu` combination so a later bisect can answer whether an XLA update changed behavior.

We did not need to carry fresh upstream code patches from that lane in this window. The practical answer was rolling back to a known-good set of pins and keeping the validation trail clean.

### NCCL and collectives

Most NCCL failures we see are topology or environment problems rather than NCCL source bugs. We carry no NCCL source patches.

One useful structural note belongs in this catalog anyway. A particular pipeline-plus-expert-parallel combination can deadlock because the topology assumptions of the pipeline schedule and expert synchronization do not line up. That looks like a low-level collectives failure until you understand the layout. The fix is not a source patch. It is an explicit compatibility gate and clear documentation.

### Megatron-Core

Megatron-Core is the largest section of the catalog because it sits on several critical boundaries at once: graph capture, fused loss paths, DSA internals, FP8 integration, and model wrappers.

The first class of issues involves CUDA-graph safety in DSA. We hit eager-style assertions and sentinel-handling branches that are harmless in normal execution and illegal during stream capture because they synchronize through `.item()`. The local fix was to gate or rewrite those checks so capture stayed legal. The upstream lesson is broader: graph-capture compatibility must be audited explicitly, even for code that looks like harmless validation.

The second class involves DSA memory behavior. One implementation materialized a large fp32 intermediate before reducing it, which was mathematically correct and operationally expensive. The local fix was to stream the reduction and accumulate directly into the smaller destination tensor. The lesson is that many memory bugs are really "wrong intermediate" bugs.

The third class involves fused loss paths. We hit a fused linear cross-entropy backward path that behaved correctly for scalar-style reductions and incorrectly for non-uniform per-token weighting. That is the kind of bug that can hide in plain sight because the default tests do not exercise masked reductions. The local workaround was to keep the fused path where it was safe and route the unsupported reduction shape through a correct wrapper.

We also hit architecture-gating issues around fused linear cross-entropy support, plus a regression where one model path no longer picked up the fused head that the adjacent GPT path already used. Those are easy bugs to miss because they often arrive as integration regressions rather than obvious math failures.

### Triton

We did not carry first-party Triton source patches in this window, but we did hit Triton-adjacent behavior worth documenting. In one kernel, a math intrinsic lowered less efficiently than expected on current nightlies, so the local implementation kept a more explicit fast path.

That is not automatically an upstream bug. Sometimes the right move is to document the limitation, use the clearer local path, and wait until the compiler contract is stable enough to justify a contribution.

### Transformer Engine

Transformer Engine sits close enough to the critical path that it deserves its own entry. On one hardware lane we saw an FP8 backward path fail an backend alignment assertion even though the architectural dimensions looked valid. The same stack passed on a different accelerator lane.

The practical fix was to limit FP8 hybrid training to the lane that we had validated and keep the other lane on BF16 smoke coverage. Not every library issue should be patched immediately. Some should be isolated, documented, and routed around until the owning project can address them properly.

### Fast Hadamard Transform

This was not a code bug, it was a packaging bug: an sdist was missing a core C++ source file. The source repository was fine, the published source package was not.

The workaround was operational rather than architectural: install from a known-good source instead of pretending the broken package belongs in the long-term patch lane. Packaging mistakes should usually live in bring-up scripts and environment notes, not in the same bucket as durable code fixes.

## How it lands in MegaCpp

Each entry above has a mirror in production. Some live as small import-time overlays or subclass overrides in the Megatron integration layer. Some live in a narrow local `mamba_ssm` fork. Some remain operational notes because the right answer is a pin, a feature gate, or a hardware-specific restriction rather than a code patch.

The storage mechanism is less important than the discipline around it. Every local fix must be idempotent where possible, must be tied to a known upstream state, and must have a visible retirement condition.

## Ablations and what we kept

The ablation question for every entry is the same: does the workaround change training numerics in a way that matters?

The DSA patches are behavior-preserving by construction. The grouped Mamba fixes are checked against supported edge cases. Precision-related fixes are compared against higher-precision references. Fused-loss workarounds are validated against non-fused baselines on the masked-loss shapes that motivated them.

What survived contact with real hardware is in this catalog. What did not make it in were one-off hacks that only hid the symptom. Nothing should enter a public patch inventory unless it is readable, testable, and worth upstreaming.

## Production checklist

- Every local patch names a corresponding upstream issue, PR, or public-ready draft.
- Every local patch has a retirement condition.
- Every patch ships with a reproducer that fails without the patch and passes with it.
- Import-time patches are idempotent and gated on the upstream state they compensate for.
- Regression guards stay alive after the upstream fix lands; deletion is explicit, not assumed.
- Packaging bugs live in environment bring-up notes, not in the long-term patch lane.
- Operational workarounds such as env flags or backend restrictions are tracked separately from real source patches.

## References

- Public upstream issues and pull requests where relevant
- Public engineering notes on DSA, sparse MLA, fused loss paths, and Mamba-3 integration
- Validation evidence, summarized in publication-safe form
