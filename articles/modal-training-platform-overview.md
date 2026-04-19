---
title: "Modal Training Platform Overview"
description: "Why we use Modal for ad-hoc training and benchmark jobs, how the image, GPU, volume, and secret model is wired, and when Modal wins against reserved H200 or TPU capacity."
date: "2026-04-18"
tags: ["modal", "training", "benchmarks", "infrastructure"]
---

We run training and benchmarks on three surfaces: reserved H200 hosts, TPU slices, and Modal. Modal is not always the cheapest per GPU-hour and it is not always the fastest to warm up, but it is unusually good at letting one engineer launch a clean, isolated job without first coordinating access to a shared machine. This post is about where that trade lands in practice: what the image, GPU, volume, and secret model looks like, and the specific regimes where Modal beats reserved capacity and where it loses.

## Why This Matters

For ad-hoc training and benchmarking, operator time is part of the cost model. A platform that takes longer to provision but makes every run clean and reproducible can be a better choice than a nominally cheaper machine that needs manual prep each time.

## The Modal Surface

The Modal surface is not "one harness." A typical setup breaks into three distinct lanes:

1. whole-model training benchmarks
2. detached backend or microbenchmark runs with explicit manifests and result collection
3. validation runs where the result is acceptance or promotion, not just throughput

In the whole-model lane, a single `modal.App` can own the image, the GPU spec, the mounted storage, and the entrypoint function. Modal's GPU configuration supports a wide range of accelerator types and counts, which means the same basic control surface can cover anything from a single-GPU smoke test to a larger multi-GPU launch.

Volumes are where Modal earns its keep. In practice, we separate at least four kinds of state:

1. a compiler-cache volume so compile-heavy jobs do not start from absolute zero every time
2. a checkpoint volume so long runs can resume cleanly
3. a local-dataset or scratch volume so preprocessed data and copied shards can survive container turnover
4. a user-home or tool-state volume for credentials, caches, and generated local artifacts that should persist across runs

That is the core Modal trick: make an ephemeral container behave just statefully enough to be useful. If you skip the volumes, every run is a cold-start experiment. If you separate them properly, Modal becomes fast enough for repeated benchmark waves while staying disposable.

Secrets are boring on purpose. Credentials should live in Modal Secrets or a cloud-provider secret manager, not in the image and not in the repository. If object storage is mounted into the job, Modal's documented storage integrations are the right way to do it.

The benchmark lanes layer on top of that. The useful pattern is detach-and-collect: submit an explicit job, keep the launch metadata, and collect structured results later instead of relying on a long interactive terminal session.

## Where Modal Wins

The practical routing rules ended up being simple:

- Modal wins for detached benchmark waves, batch validation, quick single-GPU smokes, and situations where operator time matters more than warm local state.
- Reserved H200 hosts win for tightly coupled multi-GPU training, cache-sensitive bringup, and runs where we want the machine to look similar from one day to the next.
- TPU wins when the model and runtime are already aligned with the XLA lane and the question is scale or TPU economics rather than CUDA-specific behavior.

That sounds generic, but Modal's product model makes the first line genuinely strong. Its docs are explicit about the core building blocks: GPU selection, persistent volumes, named secrets, and detached execution. Those features are enough to make isolated runs pleasant without pretending Modal is the right answer for every distributed job.

## What Changed Our Minds

We stopped describing Modal as "just another cloud GPU provider." The useful distinction is the execution model: detached jobs, persistent mounted state, explicit app lifecycle, and fast operator handoff.

We also stopped describing Modal as "ephemeral" in the simplistic sense. A Modal container is ephemeral, but the working set does not have to be. Once cache, checkpoints, and scratch space are split into separate persistent volumes, the platform behaves much more like a controlled disposable worker than like a stateless demo environment.

That does not make Modal a universal answer. For tightly coupled multi-GPU training, warm dedicated hosts still have obvious advantages. But for detached jobs and fast iteration, Modal is one of the cleanest public platforms available.

## References

- [Modal documentation](https://modal.com/docs)
- [Modal GPU guide](https://modal.com/docs/guide/gpu)
- [Modal Volumes guide](https://modal.com/docs/guide/volumes)
- [Modal Secrets guide](https://modal.com/docs/guide/secrets)
- [Modal CloudBucketMount reference](https://modal.com/docs/reference/modal.CloudBucketMount)
