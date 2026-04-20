---
title: "Modal vs Owned H200:8 vs TPU: Which Surface We Use and Why"
description: "How we decide between Modal, reserved H200:8 hosts, and TPU slices based on operator overhead, latency to first useful step, benchmark hygiene, and failure isolation."
date: "2026-04-18"
tags: ["modal", "H200", "tpu", "infrastructure", "benchmarks"]
---

We do not treat Modal, reserved H200:8 hosts, and TPU slices as interchangeable compute. They are three different operating surfaces with different strengths. Modal wins when we need a clean detached launch, fast operator turnaround, or many isolated benchmark jobs. Owned H200:8 wins when we need repeatable multi-GPU training, warm compile caches, and direct control over topology and resident state. TPU slices win when the work is already aligned with the XLA path and we care more about stable large-scale training economics than about CUDA-path kernel experiments.

## Why We Care

The wrong hardware choice wastes more time than a mediocre kernel. In practice we repeatedly run into three different job shapes:

1. long-running distributed training where compile state, checkpoint cadence, and fabric stability matter more than launch convenience
2. detached benchmark or validation waves where one engineer wants a fresh environment and a benchmark record, not fleet coordination
3. TPU-native training and data-flow work where the right question is not "can CUDA do this too" but "is the XLA lane healthy enough to keep scaling"

That separation is the same mental model we use for hardware decisions. A surface is only good if it fits the lane.

The decision is operational, not ideological. Modal is designed around containerized, detached execution and supports a broad range of GPU options, which makes it attractive for isolated runs and fast iteration. NVIDIA H200 systems are better suited to workloads that depend on stable topology, persistent local state, and tightly coupled multi-GPU behavior. TPU slices are a different lane again because they depend on the XLA stack rather than the CUDA stack.

## How The Split Works

Modal works well when the unit of work is a containerized job with explicit image, storage, and secret wiring. Its execution model is a good fit for detached benchmark waves, one-off validation runs, and other jobs where clean isolation is more important than preserving warm host state.

Owned H200:8 hosts are the opposite trade. They take more operator effort, but they make it easier to keep compile caches, datasets, checkpoints, and topology stable across repeated runs. That matters for long-running distributed training and for debugging problems that only appear once many GPUs have to move in lockstep.

TPU is the third surface. It is the right choice when the model and training stack are already aligned with PyTorch/XLA or JAX-style execution and the goal is to scale that lane rather than to test CUDA-specific kernels.

One subtle but important detail is that the runtime surface is stateful even when the compute is rented on demand. The same GPU can behave like a very different machine depending on whether caches, checkpoints, and local data are already warm. That is why we do not flatten the decision to "owned is stable, Modal is ephemeral." Modal can be made meaningfully stateful with explicit volume discipline, while the host lane tends to inherit that state more naturally.

Another detail worth keeping is that Modal gives us independent-container scale before it gives us reliable tightly coupled scale. If the work decomposes into many detached validations or many isolated benchmark cases, Modal is often the easiest and most truthful place to run it. If the work requires synchronized distributed behavior from the first compiled forward onward, owned H200:8 usually wins even before we look at cost.

The table below is the practical routing summary we keep in mind.

| Surface | Best for | Main upside | Main downside | Source of truth |
|---|---|---|---|---|
| Modal | Detached benchmark waves, validation jobs, single-GPU training smokes | Low operator overhead, reproducible fresh image, clean app lifecycle | Cold starts, image drift risk, weaker fit for tightly coupled multi-GPU work | [Modal docs](https://modal.com/docs), [GPU guide](https://modal.com/docs/guide/gpu) |
| Owned H200:8 | Multi-GPU training, compile-sensitive bringup, cache-heavy runs | Warm cache, stable topology, direct control over state and artifacts | Higher scheduling and operator burden | Local host runbooks and receipts |
| TPU | XLA-aligned training lanes, long-running scale-up work | Good fit for XLA graphs and TPU economics | Different compiler/runtime model, weaker fit for CUDA-specific experimentation | [Cloud TPU docs](https://cloud.google.com/tpu/docs), [PyTorch/XLA docs](https://docs.pytorch.org/xla/master/) |

## Practical Routing Rules

We keep benchmark and training claims surface-aware. A Modal result supports a Modal claim. A warm-host result supports a warm-host claim. A TPU result supports a TPU claim. Treating them as directly interchangeable usually hides the reason a run was fast, slow, stable, or fragile.

We also preserve provenance. For any benchmark or training note, the useful unit is not just throughput. It is the tuple of code revision, hardware surface, launch shape, runtime state, and artifact bundle. Without that context, comparisons are easy to misread.

That becomes especially important once benchmark, training, and acceptance evidence are shown side by side. If the receipt layer does not preserve which compute surface produced the result, operators eventually compare a cold detached Modal run against a warm host rerun and call the difference a regression.

It also means cost reporting needs context. Raw hourly pricing matters, but the more actionable question is usually latency to first trustworthy result. A slightly more expensive detached run can still be the right choice if it gets an engineer to a clean answer much faster.

## What Changed Our Thinking

We did not keep the idea that Modal is just "cloud H200 with prettier UX." It is a distinct execution surface with different strengths and failure modes. Modal's detached model is excellent for isolated jobs, but tightly coupled distributed work still depends more heavily on synchronized startup behavior and preserved local state.

We also did not keep a pure cost-per-GPU-hour decision rule. The better heuristic is latency to first trustworthy result. If Modal gets us a clean detached validation wave in minutes while the owned fleet is busy, it can be the cheaper choice in engineering time even when the hourly rate is worse.

One more thing we kept: cache and state are first-class. The more a lane depends on resident compile or data state, the more "compute surface" becomes a state-management question rather than a raw hardware question.

## References

- [Modal documentation](https://modal.com/docs)
- [Modal GPU guide](https://modal.com/docs/guide/gpu)
- [Modal Volumes guide](https://modal.com/docs/guide/volumes)
- [Modal Secrets guide](https://modal.com/docs/guide/secrets)
- [Modal CloudBucketMount guide](https://modal.com/docs/reference/modal.CloudBucketMount)
- [NVIDIA H200 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h200/)
- [Cloud TPU documentation](https://cloud.google.com/tpu/docs)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/master/)
