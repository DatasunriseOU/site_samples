---
title: "Profiler and performance reports: making benchmark runs comparable months later"
description: "How MegaCpp samples training, what a structured performance report should contain, and how observability stays bounded so measurement does not become the regression."
date: "2026-04-18"
tags: ["observability", "profiler", "goodput", "performance-reports", "benchmark"]
---

Reproducibility in a fast-moving training stack is mostly a paperwork problem. The model code changes, the optimizer recipe drifts, the loader contract gets tightened, the wheel of `torch` rolls forward, and six weeks later somebody tries to compare a new run to an old screenshot or chat note. The fix is not heroic profiling; it is the discipline of recording the same five things on every run, in the same format, with a known-bounded sampling cost. This post is about the reporting layer that makes those comparisons hold up over time.

## Why MegaCpp cares about this

Benchmarks rot. Specifically: a `tok/sec` number with no schema attached has a half-life of about two weeks before it stops being comparable to anything. The dataloader contract changes, the bucket alignment changes, the precision recipe changes, the kernel set changes. If we want the loss curve from week N to be honestly compared to the loss curve from week N+8, we need the run itself to carry the metadata that explains every line on the chart.

The second reason is failure attribution. Most long runs do not fail outright; they degrade. `tok/sec` falls 8% one Tuesday and nobody can say whether it was a framework bump, a communication-library change, or a layout rewrite. Structured reports make those attributions reviewable instead of vibes.

The third reason is honesty. MegaCpp publishes numbers. The numbers should be defensible. The performance report is the defense.

## What we built in MegaCpp

The instrumentation layer is intentionally small and uses Python stdlib wherever possible so it adds zero device overhead and no new hard dependencies.

The public goodput tracker is the wall-clock accountant. Adapted from the MaxText `GoodputRecorder` model and simplified for a single-leader training loop, `GoodputTracker` records named milestones (`job_start`, `tpu_init`, `training_preparation`, ...) and accumulates duration per category via the `span(category)` context manager. Categories are `step`, `checkpoint`, `eval`, `compilation`, `data_loading`. `compute_goodput()` returns `step_time / wall_time`; `compute_badput_breakdown()` returns the per-category time plus an `idle` residual that catches the noise the categories miss. The implementation is thread-safe via a single `Lock` because checkpoint saves can run on a background thread. The cost model is trivial: a `time.monotonic()` call at span entry and exit, a dict update under a lock, and a periodic dict copy when somebody calls `summary()`. Sampling cost is therefore bounded by the number of spans per step, which we keep at one (`with tracker.span("step"): forward(); backward(); step()`).

The public report builder constructs the run header. It resolves commit, branch, dirty status, and a short commit message, with explicit environment-variable overrides for rollout environments that ship without a `.git` directory. It also records GPU inventory, platform, Python version, PyTorch version, CPU count, RAM, working-directory context, and an optional cost estimate. The header is written once at job start; it is what makes the performance report still searchable six months later.

The public temporal-performance tracker is the per-task performance surface for evaluation lanes. It exposes a `start_run(config=...)`, `record_step(step, metrics, commits=, tokens=)`, `finish_run()`, and `to_json(path)` lifecycle, with a stable schema-version string baked into the file so future readers can route by version. Peak memory is read from `/proc/self/status:VmHWM` with a `resource.getrusage` fallback for macOS development. Throughput is reported as `commits_per_sec` and `tokens_per_sec`. The summary block includes mean, median, min, and max for every metric the caller recorded, computed once at `finish_run()` time so per-step recording stays cheap.

For cloud-run training, the reporting layer parses trainer stdout with a small set of compiled regexes and emits a typed result tagged with an explicit schema version. The summary block is the steady-state aggregate: throughput, loss, gradient norm, MFU, step count, training time, peak memory, and model size. The checks block is the boolean health gate: finite losses, zero exit code, presence of steps, and absence of OOM. Failure runs additionally carry a bounded reason and log tail. That is the minimum needed for a later reader to distinguish a healthy slow run from a broken fast one.

The public report schema covers ablation experiments too. The requirement is simple: every per-layer curve should match the same `steps` length, layer keys should match the declared model depth, and the model configuration should identify the architecture under test. The point of this structured design is that two ablation runs from different weeks can be opened by the same reader and compared without any ad-hoc parsing.

The public observability layer is the live-telemetry surface. Two things run by default with zero operator action: a metrics pusher sends loss, throughput, and MFU to a monitoring backend every 15 seconds, and OpenTelemetry spans wrap checkpoint saves, validation passes, and eval phases. The push interval is rate-limit-safe and leaves a generous margin. On-demand profiling should stay explicit and bounded: start it only for the window someone is debugging, stop it promptly, and store the resulting trace out of band rather than bloating the main report.

## How it lands in MegaCpp

The schema contracts should lift as-is. The trainer should write the same dict shape, the same field names, and the same float precision, and the reader should not need to know which trainer produced the file. Schema bumps should be explicit version strings, not silent additions.

The public goodput accountant also lifts cleanly. It is stdlib-only, thread-safe, and the cost model is bounded by span count per step, which should stay close to one `step` span plus small category spans at phase boundaries.

The public report builder may need two kinds of adaptation in a production deployment. First, build provenance can be injected as one structured blob by the release pipeline instead of a loose set of environment variables. Second, cost estimates should come from a centralized pricing source rather than from hand-entered values. The report should not contain guesses.

Cloud-specific parsing helpers should not become the long-term center of the system. If multiple execution surfaces exist, the parsing rules should move into a shared log-parsing layer so the same report format can be produced across environments.

The observability layer may still need selective rewrites. The monitoring push path can stay, the OpenTelemetry tracer can stay, and the on-demand profiler hooks can stay because they are a cheap “trace what is happening right now” interface. What should change is any hardcoded rate limit or label set that really belongs to a recipe or deployment preset.

The temporal-performance tracker also lifts cleanly into evaluation harnesses. The `/proc/self/status` peak-memory path is the natural Linux default, and a lightweight fallback is enough for development machines.

## Ablations and what we kept

The instrumentation surface itself has been ablated more than once. Three patterns survived; three did not.

Survived:

- A single `span("step")` per training iteration plus separate spans for `checkpoint`, `eval`, `compilation`, `data_loading`. This gives us a clean goodput number and a defensible badput breakdown without per-microbatch instrumentation.
- Stdout-parsing performance reports. The trainer prints structured step lines; the report builder parses them after the run. This means the trainer never has to know it is being measured, and an old report can be re-derived from an old log file as long as the format is stable.
- Schema versioning on every performance report. Versioned dicts beat unversioned ones every time an older file needs to be read months later.

Dropped:

- Per-microbatch tracing. The signal-to-noise ratio was poor and the sampling cost ate measurable step time on small models.
- An attempt to embed the profiler trace directly into the main report JSON. Trace files are large enough that they should be referenced by URI rather than inlined.
- A "rich report" path that recorded every CLI flag verbatim. The better approach is to record only the flags that affect numerics or performance.

The dashboards worth trusting are the ones that read the structured report rather than re-deriving from logs. The most useful panels are median `tok/sec` over time per preset, goodput fraction over time per lane, peak memory MiB over time per preset, and report-check pass rate over the last 100 runs. Anything else is supplementary.

The sample-cost budget is the discipline that keeps observability from becoming the regression it is supposed to catch. The budget on a training lane should be: at most one Python lock acquire per step for goodput accounting, at most one monitoring write every 15 seconds, zero device-side instrumentation by default, and on-demand profiling only inside an explicit bounded window. Observability cost should be measured the same way model cost is measured: if a monitoring change moves step time, the change should be reverted.

Failure-mode honesty matters too. The report `checks` block should not hardcode a throughput threshold. Throughput thresholds are recipe-dependent and versioned; a precision-stress run will look slow against a performance-tuned run, and the health block should not misclassify that as failure. The `checks` block is only the boolean health surface; performance comparison belongs at dashboard time, not at write time.

## Production checklist

- Wrap the training step in `goodput.span("step")`, checkpoint saves in `goodput.span("checkpoint")`, eval in `goodput.span("eval")`, compile warmup in `goodput.span("compilation")`, and the dataloader pull in `goodput.span("data_loading")`.
- Write the run header at job start and include git provenance, GPU info, system info, and cost info.
- Tag every performance report with an explicit schema version; do not silently add fields.
- Keep `MetricsPusher` push interval >= 15 s and respect the per-time-series rate limit.
- Keep on-demand profiling behind `SIGUSR1` / `SIGUSR2` and never enable it by default on a long run.
- Reference profiler-trace files by URI in the report; do not inline them.
- Validate every ablation report against its schema before writing.
- Run the dashboard against the report store, not against raw logs.
- Treat any change that moves step time as an observability cost regression and bisect it before merging.
- Persist performance reports to durable storage near the checkpoint, so a recovered checkpoint always has its matching report.

## References

- https://github.com/AI-Hypercomputer/maxtext
- https://opentelemetry.io/docs/specs/otel/
- https://cloud.google.com/monitoring/quotas
- https://research.google/pubs/the-tail-at-scale/
