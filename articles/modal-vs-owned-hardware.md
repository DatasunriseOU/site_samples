---
title: "Modal vs Owned H200:8 vs TPU: Which Surface We Use and Why"
description: "How MegaCpp decides between Modal, reserved H200:8 hosts, and TPU slices, based on operator overhead, latency to first useful step, benchmark hygiene, and failure isolation."
date: "2026-04-18"
tags: ["modal", "h200", "tpu", "infrastructure", "benchmarks"]
---

TL;DR: we do not treat Modal, reserved H200:8 hosts, and TPU slices as interchangeable compute. The research repo keeps them as three operating surfaces because they optimize for different things. Modal wins when we need a clean detached launch, fast operator turnaround, or many isolated benchmark jobs. Owned H200:8 wins when we need repeatable multi-GPU training, warm compile caches, and direct control over topology and resident state. TPU slices win when the work is already aligned with the XLA path and we care more about stable large-scale training economics than about CUDA-path kernel experiments.

## Why MegaCpp cares

The wrong hardware choice wastes more time than a mediocre kernel. In the research repo we repeatedly run into three different job shapes:

1. long-running distributed training where compile state, checkpoint cadence, and fabric stability matter more than launch convenience
2. detached benchmark or validation waves where one engineer wants a fresh environment and a benchmark record, not fleet coordination
3. TPU-native training and data-flow work where the right question is not "can CUDA do this too" but "is the XLA lane healthy enough to keep scaling"

The benchmark planning notes are useful here because they explicitly warn against collapsing every Modal workflow into one generic story. They separate whole-model training benchmarks, exact-token sparse detached benchmarks, and sparse validation and promotion. That separation is the same mental model we use for hardware decisions. A surface is only good if it fits the lane.

The decision is also operational, not ideological. The multi-GPU status notes say single-GPU runs are healthy on multiple GPU types, while the 8-GPU FSDP2 plus compile lane still hangs on Modal when ranks diverge during cold Triton compilation. That is enough to change routing. Once a lane depends on warm compile outputs and synchronized startup behavior, our owned H200:8 hosts stop being a luxury and become the source of truth.

## What we built in the POC

The POC already encodes this split in code.

The main training entrypoint and benchmark matrix define the Modal whole-model lane. They package a detached app, GPU selection, volumes, secrets, and per-case metadata so one engineer can launch a controlled wave without logging into a host first. The detached launcher and collector define a second Modal lane built around explicit manifests, call identity, backend identity, and remote runtime provenance. The sparse validation launcher defines a third lane where success is promotion status and pass/fail summaries rather than raw throughput.

At the same time, the owned-host path stays central in the docs. The multi-GPU status notes are blunt that the multi-GPU training lane works on warm-cache H200 hosts and is still being stabilized on Modal. The reason is practical: when compile-heavy ranks advance at different speeds, the first distributed collective can deadlock. On a host where the cache already exists, that startup shape is materially different.

TPU is the third surface. The site already has posts such as `fsdp2-on-xla-tpu.md`, `tokenized-enriched-pipeline-on-tpu.md`, and `torch-xla-pjrt-reality.md`, and the repo-side training/report machinery is built to carry provenance across environments instead of pretending they are identical. `report.py` records git identity, GPU visibility, host details, and optional cost estimates. That matters because our comparison unit is not just tok/sec. It is "what exactly ran, where, with what resident state, and under which launch contract".

One subtle but important detail from the main training entrypoint is that the runtime surface is stateful even when the compute is rented on demand. The code treats checkpoint state, inductor cache, and copied local dataset state as separate resources, and it does so because the same GPU can behave like a very different machine depending on which of those are warm. That is also why we do not flatten the decision to "owned is stable, Modal is ephemeral". Modal can be made meaningfully stateful; it just requires explicit volume discipline, while the host lane tends to inherit that state more naturally.

Another detail worth keeping is that Modal gives us independent-container scale before it gives us reliable tightly coupled scale. That sounds obvious, but it is one of the main routing heuristics we use in practice. If the work decomposes into many detached validations or many isolated benchmark cases, Modal is often the easiest and most truthful place to run it. If the work requires synchronized distributed behavior from the first compiled forward onward, owned H200:8 usually wins even before we look at cost.

The table below is the practical routing summary we keep in mind.

| Surface | Best for | Main upside | Main downside | Source of truth |
|---|---|---|---|---|
| Modal | Detached benchmark waves, sparse acceptance, single-GPU training smokes | Lowest operator overhead, reproducible fresh image, clean app lifecycle | Cold starts, image drift risk, weaker multi-GPU confidence when cache state matters | `modal_train.py`, `modal_matrix.py`, `modal_bench_dsa_backend_detach.py`, `modal_sparse_validation_detach.py` |
| Owned H200:8 | Multi-GPU training, compile-sensitive bringup, cache-heavy runs | Warm cache, stable topology, direct control over state and artifacts | Higher scheduling and operator burden | `MODAL_MULTI_GPU_STATUS.md`, current training receipts, host-side reports |
| TPU | XLA-native training and data-flow experiments | Strong fit for the XLA stack, stable large-run lane when the graph is already there | Not the place for CUDA-path kernel questions | TPU-focused posts plus shared report/provenance surfaces |

## How it lands in MegaCpp

MegaCpp inherits the policy, not just the launchers.

First, we keep benchmark and training claims surface-aware. That comes directly from `MODAL_BENCHMARK_PLAN.md`, which says old JSON artifacts are dated evidence, not blanket proof that the current lane is healthy. In MegaCpp terms, that means a Modal receipt is enough to justify a Modal claim, but not enough to rewrite the H200:8 training story unless the lane matches.

Second, we preserve provenance. `report.py` gathers git commit, branch, dirty state, host metadata, GPU inventory, and optional cost estimates. Even though its default hourly-rate table is intentionally rough, the point is solid: every receipt should carry enough context to explain itself later. A benchmark number without launch shape is not a result, it is a rumor.

Third, we keep observability and receipts separate from marketing claims. `observability.py` is built around artifact durability: it defines cloud upload policy, artifact manifests, local staging, and explicit categories like logs, traces, flamegraphs, reports, and summaries. That is the backend contract we want in MegaCpp too. If a Modal run produced a useful comparison, we want the report, summary, and metadata bundle, not just the stdout excerpt someone pasted into chat.

Fourth, we use the same rule for owned hosts. A host run is not automatically more trustworthy because we own the machine. It is more trustworthy when it has the receipt discipline that Modal forced us to build anyway. `modal_receipts.py` normalizes call IDs, status, timestamps, summaries, and error payloads; the broader lesson is that we should normalize owned-host receipts to the same standard whenever possible.

That becomes especially important once MegaCpp starts mixing training, benchmark, and acceptance evidence in one reporting surface. If the receipt layer does not preserve which compute surface produced the result, operators will eventually compare a cold detached Modal run against a warm host rerun and call the difference a regression. The cure is boring but effective: surface identity has to stay in the result, and comparison tools have to refuse weak matches.

It also means cost reporting needs context. `report.py` can estimate cost, but the more actionable output for operators is usually a joint view of cost, startup shape, and artifact quality. We care whether a run was cheap, but we care more whether it was cheap while still being reproducible and comparable. A benchmark that saved a few dollars and lost its provenance is not a win.

## Ablations and what we kept

Several tempting simplifications did not survive contact with the code and docs.

We did not keep the idea that Modal is just "cloud H200 with prettier UX". The POC disproves that. `MODAL_MULTI_GPU_STATUS.md` documents a specific failure mode around cold compile divergence across ranks. That is not a cosmetic issue; it changes where we trust multi-GPU results.

We did not keep the idea that all Modal receipts mean the same thing. `MODAL_BENCHMARK_PLAN.md` makes lane-specific bookkeeping mandatory. Whole-model training wants steady-state step metrics and exact distributed mode. Exact-token sparse wants runtime telemetry, backend identity, and collector state transitions. Validation wants promotion summaries. Mixing them erases the whole point of having receipts.

We also did not keep a pure cost-per-GPU-hour decision rule. `MODAL_MULTI_GPU_STATUS.md` includes per-GPU pricing, and `report.py` has cost-estimation hooks, but those are inputs, not the full answer. The better heuristic is latency to first trustworthy result. If Modal gets us a clean detached validation wave in minutes while the owned fleet is busy, it can be the cheaper choice in engineering time even when the hourly rate is worse.

One more thing we kept: cache and state are first-class. `modal_train.py` treats checkpoint state, inductor cache, and local copied data as separate mounted volumes for a reason. That same separation is why owned hosts remain valuable. The more a lane depends on resident compile or data state, the more "compute surface" becomes a state-management question rather than a raw hardware question.

We also kept a strong bias toward reversible routing. If an experiment starts on Modal and the receipt shows the wrong startup shape, we would rather move it to owned H200:8 than spend hours trying to prove a point about parity. Likewise, if a question is really about TPU-scale training behavior, the right move is often to stop arguing with the CUDA lane and move the experiment where the graph actually lives. Good routing is a way to avoid debugging the wrong infrastructure.

Finally, we kept the operator-centered perspective. The surface choice should reduce coordination cost for the people running the work. That is why Modal remains important even with owned hosts available. It shortens the path from idea to detached evidence for many benchmark and validation tasks. The goal is not to crown one winner. The goal is to keep each surface doing the jobs it is structurally good at.

Here is the kind of routing rule we actually want engineers to follow:

```text
example platform split:
  use Modal for detached sparse acceptance and single-GPU benchmark waves
  collect a manifest-backed receipt after the run
  use owned H200:8 machines for compile-sensitive multi-GPU bring-up
  use TPU when the experiment is already on the XLA path and the goal is TPU-scale training
```

## Production checklist

- Route by lane, not by brand name. Start from the job shape in `MODAL_BENCHMARK_PLAN.md`.
- Use Modal when clean detached execution and operator speed matter more than resident cache state.
- Prefer owned H200:8 for multi-GPU training when compile warmup and synchronized startup behavior are part of correctness.
- Treat TPU as its own production lane with its own graph, sharding, and observability assumptions.
- Require provenance on every receipt: commit, launch flags, runtime surface, and artifact bundle.
- Compare steady-state metrics, not step-0 startup noise.
- Never upgrade a claim from one surface to another without matched receipts.
- Preserve state intentionally: cache, checkpoints, copied data, manifests, logs, and summaries.

## References

- a Modal benchmark planning note
- a Modal multi-GPU status note
- the public Modal training launcher sample
- the public Modal benchmark matrix sample
- the detached benchmark launcher and collector samples
- the sparse validation launcher sample
- the shared reporting and receipt utilities
- the observability utilities
