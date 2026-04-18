---
title: "Modal Debugging Guide for Training and Benchmark Failures"
description: "A grounded guide for debugging Modal failures in the MegaCpp POC: cold starts, multi-GPU hangs, image drift, detached collector issues, and volume or output-state bugs."
date: "2026-04-18"
tags: ["modal", "debugging", "benchmarks", "training", "observability"]
---

TL;DR: when a Modal run fails, we do not start by staring at stdout and guessing. We classify the failure by lane, then by lifecycle stage: image/bootstrap, detached launch, remote execution, collector state, benchmark-record generation, or output persistence. The POC already has the right surfaces for that workflow. The benchmark planning notes tell us which lane we are in, the multi-GPU status notes document the known multi-GPU failure mode, the benchmark-record utilities and collector scripts expose detached state transitions, and the observability layer defines how logs and outputs should survive long enough to debug.

## Why MegaCpp cares

Training failures are expensive, but ambiguous failures are worse. A clean error that says "image missing dependency" costs us minutes. A detached Modal app that appears to launch, hangs later, and leaves partial logs with no manifest update can waste a day.

The POC makes this manageable because it already separates surfaces. The benchmark plan says we have three Modal lanes, each with different success criteria and bookkeeping. That means the first debugging question is not "why did Modal fail". It is "which contract failed".

That distinction matters because the remedies differ:

- whole-model training failures usually involve startup state, volumes, distributed mode, or steady-state metric parsing
- exact-token sparse detached failures usually involve manifest lifecycle, backend identity, or remote runtime provenance
- sparse validation failures are often promotion-status or backend-bootstrap problems rather than throughput problems

Once we debug at the contract boundary, the failures stop looking random.

## What we built in the POC

The POC already has the pieces of a real debugging guide.

`modal_train.py` is the launch surface for whole-model training. It wires the app, image, GPU choice, mounted volumes, and runtime environment. That makes it the first place to look for cold-start or image drift issues. If the run never reaches useful work, the failure is often in this bootstrap layer.

`modal_matrix.py` adds structure around that launch path by naming cases, setting flags, detaching execution, and recording per-case outputs. If one case regresses and the others do not, the matrix itself becomes the smallest useful repro.

For detached sparse runs, `modal_bench_dsa_backend_detach.py` creates the remote call and writes a manifest, while `modal_bench_dsa_backend_collect.py` advances that manifest through `detached`, `running`, `ok`, or `error`. The collector also snapshots call-graph state on timeouts and errors, and it writes `bench_result`, `bench_telemetry`, `backend_identity`, `remote_output_json`, and `remote_runtime_provenance` once the remote result lands. That is already a debugging state machine.

`modal_receipts.py` and `read_modal_receipts.py` provide the readback side. They normalize receipt shape, timestamps, run identity, summaries, and error fields so we can compare runs without reverse-engineering each file by hand.

`observability.py` covers the persistence side. It is not just logging. It defines structured artifact categories, manifest generation, staged local writes, and cloud upload behavior for things like reports, traces, summaries, and logs. When debugging feels impossible, it is usually because this layer was underused or bypassed.

`report.py` matters here too, even though it reads more like benchmarking infrastructure than debugging infrastructure. In practice, a surprising number of "training failures" turn out to be provenance failures: wrong branch, dirty checkout, different visible GPU set, different machine shape, or a changed environment that nobody wrote down. Once the report layer records that information, a whole class of phantom regressions disappears.

The other useful property of the current code is that it gives us natural cut points. If the manifest exists but never advances beyond `detached`, the problem is usually launch or remote scheduling. If the manifest reaches `running` but not `ok`, the collector and call graph become the next stop. If the receipt exists but the artifact bundle is thin, the issue is no longer execution but observability discipline. Those are much better failure categories than "Modal is flaky".

## How it lands in MegaCpp

The debugging posture we want in MegaCpp is straightforward: classify first, then inspect the right artifact.

Here is the operational table we follow.

| Symptom | Likely layer | First files to inspect | Typical fix direction |
|---|---|---|---|
| Long startup before any useful step | image/bootstrap or cold cache | `modal_train.py`, `modal-image-and-cold-start.md` | bake or pin the image, reduce bootstrap drift, preserve caches intentionally |
| 8-GPU run hangs on first forward or first collective | distributed compile divergence | `MODAL_MULTI_GPU_STATUS.md`, training receipts, launch flags | move back to owned H200:8, warm compile state, avoid treating Modal as current truth for that lane |
| Detached run exists but collector never finishes | manifest lifecycle or remote call state | `modal_bench_dsa_backend_detach.py`, `modal_bench_dsa_backend_collect.py`, `modal_receipts.py` | verify `function_call_id`, poll state, inspect call graph, capture error payload |
| Result lands but numbers are suspect | lane mismatch or wrong bookkeeping | `MODAL_BENCHMARK_PLAN.md`, `report.py` | compare only within the same lane and same metric contract |
| Logs exist but no durable report bundle | artifact persistence gap | `observability.py`, generated manifests | ensure artifact categories, summary files, and uploads are enabled and committed |
| Re-run behaves differently from prior run | image drift or volume/state drift | image definition, mounted volumes, receipt provenance | pin image inputs, reload/commit volumes intentionally, compare provenance fields |

That table is grounded in the code. The collector explicitly records `last_polled_at`, call-graph snapshots, `completed_at`, and typed `error` strings. `report.py` records git, host, runtime, and hardware metadata. `observability.py` gives artifacts names and categories. This is already enough to stop most "it just hung" debugging from turning into folklore.

The same structure should shape our runbooks in MegaCpp. Instead of a single generic troubleshooting page, we want a small set of decision trees tied to receipt type. Detached sparse runs should send people toward manifest and backend-identity inspection. Training regressions should send them toward steady-state metrics, launch flags, and stateful volumes. Artifact gaps should send them toward the observability layer rather than back toward the launcher.

That sounds procedural, but it is really about compressing time to root cause. Most benchmark teams lose hours because they repeat the wrong first five steps. The POC is already opinionated enough that we do not need more theory. We just need to keep the investigation aligned with the contracts the code already exposes.

## Ablations and what we kept

The first ablation we rejected is treating all hangs as NCCL bugs. `MODAL_MULTI_GPU_STATUS.md` is more specific: the known failing lane is 8-GPU FSDP2 plus compile with cold inductor state, where ranks spend different amounts of time in Triton compilation and then deadlock at collectives. That is a different class of bug than a generic network failure, and the workaround is different too.

The second ablation we rejected is using only stdout as the debugging surface. Detached Modal execution breaks that habit. If the accepted contract is `app.run(detach=True)` plus a collector, then the manifest is not optional bookkeeping. It is part of the debugging interface. `modal_bench_dsa_backend_collect.py` proves this by storing state transitions and preserving remote-output payloads even when the remote path is inconvenient to inspect live.

The third ablation we rejected is assuming a fresh image is automatically good. Modal makes image drift easier to hide because launching is so convenient. If a dependency, wheel, or runtime patch changed between runs, the right response is to compare provenance and artifact bundles, not to assume the benchmark regressed. That is why `report.py` and the receipt surfaces matter.

The fourth ablation we rejected is using one shared volume for everything. The training lane separates checkpoint state, compile cache, and data-locality state. That turns debugging from "something in storage is weird" into a narrower question: did the checkpoint persist, did the cache warm, did the copied dataset exist, did the right run name land in the benchmark record.

What we kept is a very explicit stage-based playbook.

1. Identify the lane from the benchmark planning notes.
2. Decide whether the failure is bootstrap, remote execution, collector, benchmark record, or output persistence.
3. Read the manifest and benchmark record before re-running.
4. Compare provenance, not just metrics.
5. Only then decide whether to retry on Modal, switch to owned H200:8, or move the question to TPU.

We also kept a bias toward explaining failures in terms of state transitions instead of one-off anecdotes. If the same bug can be phrased as "collector stayed in running because remote result never satisfied expected contract" or "a benchmark record existed but the summary never uploaded", it becomes fixable by another engineer later. If it stays as "that one weird Modal hang from Tuesday", it dies as tribal knowledge.

And we kept the rule that a clean no-go verdict is often the fastest fix. The multi-GPU status notes already give us a known boundary where owned H200:8 is the safer lane. The guide is not supposed to prove Modal can do everything. It is supposed to tell us quickly whether the current failure belongs to launch hygiene, manifest handling, output persistence, or a lane we should move elsewhere.

That last point is worth emphasizing because it changes team behavior. A good debugging playbook is not only a way to fix runs; it is a way to avoid burning credibility on bad evidence. Once a lane has crossed the threshold where warm cache, synchronized compile, or host-resident state are part of correctness, the fastest route to truth is often to stop retrying on Modal and reproduce on the owned surface instead. The playbook should make that downgrade feel normal, not like defeat.

The following commands reflect that discipline:

```text
example Modal debugging flow:
  1. Read the latest benchmark record from durable storage.
  2. Compare two named runs when a regression is suspected.
  3. Validate detached collector state from a saved manifest.
  4. Check active apps when the run appears stuck between detach and the benchmark record.
```

## Production checklist

- Start every investigation by naming the lane: training, exact-token sparse, or validation.
- Do not compare metrics across lanes with different bookkeeping contracts.
- For multi-GPU hangs, check whether the failure matches the documented cold-compile divergence mode before blaming the network stack.
- Treat detached manifests as primary debugging artifacts, not side files.
- Persist reports, summaries, traces, and logs through the observability layer.
- Compare image and runtime provenance before calling a change a regression.
- Separate checkpoint, cache, and data-locality state so volume bugs are diagnosable.
- Prefer a smaller matrix repro over a blind full-wave re-run.
- When Modal is no longer the right debug surface, move the lane back to owned H200:8 instead of forcing parity by wishful thinking.

## References

- benchmark planning notes
- multi-GPU status notes
- the main training entrypoint
- `modal_matrix.py`
- `modal_bench_dsa_backend_detach.py`
- `modal_bench_dsa_backend_collect.py`
- benchmark record readers
- `check_modal_apps.py`
- benchmark record utilities
- `observability.py`
- `report.py`
