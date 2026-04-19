---
title: "Torch XLA and PJRT reality: what actually matters"
description: "A grounded look at the current TPU stack: PJRT contracts, SPMD setup order, reduction semantics, and the failure modes that still shape training and evaluation."
date: "2026-04-18"
tags: ["torch-xla", "pjrt", "xla", "tpu", "training", "evaluation"]
---

The current TPU lane is not a generic "install XLA and go" setup. It depends on the modern PJRT runtime, early SPMD initialization, and careful reduction behavior in evaluation code. The practical rule is simple: use a frontend and runtime that agree on the same contract, set runtime policy before imports, enable SPMD before tensors exist, and never assume a TPU metric is globally reduced unless the code path proves it.

The substrate boundary matters. PyTorch/XLA is the PyTorch path for TPU. JAX is
a separate frontend with its own tracing and execution model. Pallas is a JAX
kernel surface. None of those should be collapsed into an NVIDIA precision or
CUDA-kernel story.

There is a lot of stale advice around Torch XLA. Some of it was valid for older XRT-era setups, some of it assumes stock wheels are enough, and some of it ignores how training code mixes evaluation, optimizer state, and mesh construction. The decisive details are runtime ownership, import order, graph stability, and reduction semantics.

## The first reality: the stack is a runtime contract

PyTorch/XLA has moved from the older XRT runtime to PJRT, and current TPU guidance assumes PJRT by default when no older runtime is configured. That makes the version boundary between `torch`, `torch_xla`, and the TPU runtime layer part of the execution contract rather than a background packaging detail.

That explains most TPU confusion. When people say "XLA is unstable," they often mean they mixed a current TPU runtime with an older software contract. When they say "PJRT changed everything," they are partly right, but the practical lesson is operational rather than philosophical: keep the frontend and runtime on the same contract boundary.

Public Torch XLA guidance reinforces the same picture from another angle: frontend tracing, PJRT runtime behavior, and device-runtime compatibility have to align before model-level debugging even starts.

| Question                                   | Public answer                               | Why it matters                                           |
| ------------------------------------------ | ------------------------------------------- | -------------------------------------------------------- |
| Is PJRT the current runtime surface?       | Yes                                         | runtime expectations start there                         |
| Is TPU support bolt-on?                    | No                                          | initialization order and mesh semantics are central      |
| Can I trust a local TPU metric by default? | No, only after the reduction path proves it | silent fallback can misreport eval quality or throughput |

If you take only one thing from this, make it this: TPU stability starts with a correct contract between the framework frontend, PJRT runtime, and TPU runtime layer.

## The second reality: import order and SPMD timing are part of correctness

The training startup path is explicit about setup order. Runtime flags must be applied before importing `torch_xla`, and `xr.use_spmd()` must be called before any Torch XLA tensor exists. This is not cosmetic startup sequencing. It is part of the runtime contract.

That means TPU setup has two early gates.

1. Set the environment and runtime policy before importing the runtime.
2. Enable SPMD before constructing tensors or letting helper paths import XLA indirectly.

A compilation cache belongs in that same startup contract. Caching is not just a convenience; it changes how repeated runs behave and helps separate cold compile cost from actual execution regressions.

```text
example TPU startup contract:
  set PJRT_DEVICE = TPU
  apply TPU runtime flags before importing the runtime
  enable SPMD before tensors exist
  start training with an explicit runtime profile
```

The exact launcher can vary, but the principle does not. If a helper imports XLA too early, the rest of the run becomes hard to reason about.

## The third reality: evaluation can lie if reduction semantics are weak

One of the most useful surfaces for understanding TPU behavior is the evaluation path used for loss aggregation. Any metric that depends on a collective has to prove that the collective really happened. Otherwise a run can look healthy while reporting local rather than global totals.

This is not a niche bookkeeping error. In practice it affects how you interpret evaluation curves, cost-per-token calculations, and any claim about TPU scaling efficiency. If a global metric silently degraded into a local metric, then the dashboard is not merely noisy; it is wrong.

For operators, the rule should stay simple: if the metric depends on a collective, confirm the collective path explicitly.

## Mesh construction is the real TPU mental model

The right way to think about PJRT in this stack is not as a magical optimizer. It is the runtime contract underneath the TPU execution model. The engineering task is to build the right mesh, shard the right tensors, and preserve those assumptions through the training and eval stack.

That matters especially in a project that mixes different training and evaluation paths. Two TPU lanes may both use XLA and PJRT, but they may not be testing the same execution surface. The runtime contract may be global, but the failure surfaces are still local.

## The setup story is mature enough to be useful, but not simple enough to ignore

A good sign is that the public documentation is concrete here. The runtime and PJRT notes define ownership and setup order clearly. The XLA profile sample keeps policy visible instead of hiding it in shell history.

The mature posture is therefore neither optimism nor panic. It is to treat TPU startup, reduction semantics, and mesh ownership as first-class engineering surfaces rather than background details.

## References

- https://docs.pytorch.org/xla/master/runtime.html
- https://docs.pytorch.org/xla/master/learn/pjrt.html
- https://docs.jax.dev/en/latest/pallas/quickstart.html
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/xla/xla_flag_profile.py
