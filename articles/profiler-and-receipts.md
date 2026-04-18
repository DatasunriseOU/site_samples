---
title: "Profiler and receipts: making benchmark runs comparable months later"
description: "How MegaCpp samples training, what the receipt artifact contains, the dashboards we trust, and the sample-cost budget we keep so observability never becomes the regression."
date: "2026-04-18"
tags: ["observability", "profiler", "goodput", "receipts", "benchmark"]
---

Reproducibility on a fast-moving research stack is mostly a paperwork problem. The model code changes, the optimizer recipe drifts, the loader contract gets tightened, the wheel of `torch` rolls forward, and six weeks later somebody tries to compare a new run to a number on a slack screenshot from before the holiday. The fix is not heroic profiling; it is the discipline of recording the same five things on every run, in the same format, with a known-bounded sampling cost. This post is about the layer that does that for us: `goodput.py`, `report.py`, `temporal_perf.py`, `modal_receipts.py`, `receipt_schema.py`, and `observability.py`.

## Why MegaCpp cares about this

Benchmarks rot. Specifically: a `tok/sec` number with no schema attached has a half-life of about two weeks before it stops being comparable to anything. The dataloader contract changes, the bucket alignment changes, the precision recipe changes, the kernel set changes. If we want the loss curve from week N to be honestly compared to the loss curve from week N+8, we need the run itself to carry the metadata that explains every line on the chart.

The second reason is failure attribution. Most of our long runs do not fail outright; they degrade. `tok/sec` falls 8% one Tuesday and nobody can say whether it was a torch nightly bump, a NCCL plugin shuffle, or the new packed-rows producer landing a column reorder. Receipts make those attributions reviewable instead of vibes.

The third reason is honesty. We publish numbers. The numbers should be defensible. The receipt is the defense.

## What we built in the POC

The instrumentation layer is intentionally small and uses Python stdlib wherever possible so it adds zero device overhead and no new hard dependencies.

The public goodput tracker is the wall-clock accountant. Adapted from the MaxText `GoodputRecorder` model and simplified for our single-leader training loop, `GoodputTracker` records named milestones (`job_start`, `tpu_init`, `training_preparation`, ...) and accumulates duration per category via the `span(category)` context manager. Categories are `step`, `checkpoint`, `eval`, `compilation`, `data_loading`. `compute_goodput()` returns `step_time / wall_time`; `compute_badput_breakdown()` returns the per-category time plus an `idle` residual that catches the noise the categories miss. The whole file is thread-safe via a single `Lock` because checkpoint saves run on a background thread on the production lane. The cost model is trivial: a `time.monotonic()` call at span entry and exit, a dict update under a lock, and a periodic dict copy when somebody calls `summary()`. Sampling cost is therefore bounded by the number of spans per step, which we keep at one (`with tracker.span("step"): forward(); backward(); step()`).

The public report builder constructs the run header. `get_git_info()` resolves commit, branch, dirty status, and the first 80 chars of the commit message — with explicit env-var overrides (`MEGACPP_GIT_COMMIT`, `MEGACPP_GIT_BRANCH`, `MEGACPP_GIT_DIRTY`, `MEGACPP_GIT_MESSAGE`) for VM rollouts that ship without a `.git` directory. `get_gpu_info()` enumerates `torch.cuda.device_count()` with `torch.cuda.get_device_properties(i)` for names and memory; `get_system_info()` records platform, python version, torch version, CPU count, RAM, and working-directory context. `CostInfo` records hourly rate, GPU type, and an optional total estimate. The header is written once at job start; it is what makes the receipt grep-able six months later.

The public temporal-performance tracker is the per-task perf surface for evaluation lanes (commit-to-task, ripple retrieval, edit localization). `TemporalPerfTracker` exposes a `start_run(config=...)`, `record_step(step, metrics, commits=, tokens=)`, `finish_run()`, and `to_json(path)` lifecycle, with a stable schema-version string baked into the file so future readers can route by version. Peak memory is read from `/proc/self/status:VmHWM` with a `resource.getrusage` fallback for macOS dev. Throughput is reported as `commits_per_sec` and `tokens_per_sec`. The summary block includes mean/median/min/max for every metric the caller recorded, computed once at `finish_run()` time so per-step recording stays cheap.

The public Modal receipt builder is the production-run receipt surface for the Modal lane. After a run completes, `build_training_receipt()` parses the trainer's stdout with a small set of compiled regexes (`_STEP_RE` matches `step N/M ... loss: X | dt: Yms | tok/sec: Z | mfu: W`, `_GNORM_RE` and `_PEAK_MEM_RE` cover the rest), and emits a typed dict tagged with an explicit `RECEIPT_SCHEMA_VERSION` constant. The summary block is the steady-state aggregate: `mean_tok_sec`, `median_tok_sec`, `min/max_tok_sec`, `final_loss`, `first_loss`, `mean_loss`, `max_gnorm`, `mean_gnorm`, `mean_mfu`, `total_steps`, `steady_state_steps`, `training_time_sec`, `peak_memory_mib`, `num_params`. The checks block is the boolean health gate: `all_losses_finite`, `no_nan_loss`, `exit_code_zero`, `has_steps`, `no_oom`. Failure runs additionally carry `failure_reason` and a bounded `log_tail`. The receipt is written to the checkpoint volume so it can be retrieved later via the Modal volume CLI.

The public receipt schema covers the ablation-experiment receipt surface. `SinkAblationReceipt` and `OutlierAblationReceipt` are dataclasses with explicit `validate()` methods that enforce shape invariants — every per-layer curve has the same length as `steps`, layer keys cover `range(model_config["n_layer"])`, and `model_config` is required to identify the architecture under test. Both serialize to JSON via `to_json()` / `from_json()`, with int layer keys converted to strings on the way out and back on the way in. The point of the structured-receipt design is that two ablation runs from different weeks can be opened by the same reader and compared without any ad-hoc parsing.

The public observability layer is the live-telemetry surface. Two things run by default with zero operator action: `MetricsPusher` pushes loss / throughput / MFU to Cloud Monitoring every 15 seconds, and OpenTelemetry Cloud Trace spans wrap checkpoint saves, validation passes, and eval phases. The push interval is rate-limit-safe: Cloud Monitoring permits 1 point per 5 s per time series, so 15 s leaves a generous margin. On-demand profiling rides on signals: `SIGUSR1` starts an XLA profiler trace via `torch_xla.debug.profiler.start_trace(logdir)` and writes a small JSON status file under `/tmp` so the data browser can read it over SSH; `SIGUSR2` stops the trace and uploads the xplane artifact to the configured object-store bucket on a background thread. The default-on metrics path costs roughly one Cloud Monitoring write per 15 seconds and a handful of dict updates; the on-demand profiler is opt-in for exactly the windows somebody is debugging.

## How it lands in MegaCpp

The receipt schemas lift as-is. `RECEIPT_SCHEMA_VERSION` and `TEMPORAL_PERF_SCHEMA_VERSION` are stable contracts; the production trainer writes the same dict shape, the same field names, and the same float precision, and the production reader does not have to know which trainer produced the file. Schema bumps are explicit version strings, not silent additions, and the validators in `receipt_schema.py` are imported into the MegaCpp receipt-reading utilities unchanged.

The public goodput accountant lifts as-is. It is stdlib-only, thread-safe, and the cost model is bounded by span count per step, which the production trainer keeps at exactly one for `step` plus zero or one for `checkpoint`/`eval`/`compilation` per phase boundary.

The public report builder is rewritten in two specific ways on the production path. First, the `MEGACPP_GIT_*` env-var override family becomes a single `MEGACPP_BUILD_PROVENANCE` JSON blob written by the build pipeline; production VMs do not have a `.git` directory and we do not want six independent env vars to keep in sync. Second, `CostInfo.hourly_rate` is sourced from a centralized billing config rather than passed in by hand; the receipt should not contain a guess.

`modal_receipts.py` is mostly retired. The Modal lane is one of several execution surfaces in production, and we lift the parsing regex set into a shared `runtime/log_parsing.py` module so the same parser is used by the on-prem H200 lane, the TPU pod lane, and any operator-launched bare-metal run. The Modal-specific volume-write helper stays where it is.

`observability.py` is partially rewritten. The Cloud Monitoring push path stays; the OpenTelemetry tracer stays; the SIGUSR1/SIGUSR2 profiler hooks stay because they are the cheapest possible "trace what is happening right now" UI. What changes is the `MetricsPusher` rate-limit and label set, which becomes a recipe choice rather than a code constant — different presets push different label sets and we want the receipt to know which.

`temporal_perf.py` lifts as-is into the eval harness. The peak-memory `/proc/self/status` path is the canonical one; the `resource.getrusage` macOS fallback stays for the developer lane.

## Ablations and what we kept

The instrumentation surface itself has been ablated more than once. Three patterns survived; three did not.

Survived:

- A single `span("step")` per training iteration plus separate spans for `checkpoint`, `eval`, `compilation`, `data_loading`. This gives us a clean goodput number and a defensible badput breakdown without per-microbatch instrumentation.
- Stdout-parsing receipts. The trainer prints structured step lines; the receipt builder parses them after the run. This means the trainer never has to know it is being measured, and we can re-derive a receipt from an old log file as long as the format is stable.
- Schema versioning on every receipt artifact. The training-receipt schema, the temporal-perf schema, plus the per-experiment schemas in `receipt_schema.py` all carry an explicit version string. Versioned dicts beat unversioned ones every single time we have had to read a six-month-old file.

Dropped:

- Per-microbatch tracing. The signal-to-noise ratio was poor and the sampling cost ate measurable step time on small models.
- An attempt to embed the profiler-trace artifact directly into the receipt JSON. xplane files are large enough that we now reference them by URI in the receipt rather than inline.
- A "rich receipt" path that recorded every CLI flag verbatim. We now record only the flags that affect numerics or perf — flag normalization happens before write.

The dashboards we trust are the ones that read the receipt rather than re-deriving from logs. The four panels we use day-to-day are: median `tok/sec` over time per preset, goodput fraction over time per lane, peak memory MiB over time per preset, and the receipt-checks pass rate over the last 100 runs. Anything else is supplementary. If the four panels all agree the run is healthy and a fifth panel disagrees, the fifth panel is wrong until proven otherwise.

The sample-cost budget is the discipline that keeps observability from becoming the regression it is supposed to catch. The budget on the training lane is: at most one Python lock acquire per step for the goodput accountant, at most one Cloud Monitoring write per 15 seconds, zero device-side instrumentation by default, and on-demand profiling only inside an explicit signal-bracketed window. We measure observability cost the same way we measure model cost — if a `MetricsPusher` change makes step time move, the change is reverted.

Failure-mode honesty: the receipt's `checks` block deliberately does not check `mean_tok_sec >= some_threshold`. Throughput thresholds are recipe-dependent and recipe-versioned; a receipt from a precision-stress run will look slow against a perf-tuned run, and we do not want the check block to flag it as a failure. The `checks` block is only the boolean health surface; the perf comparison happens at dashboard time, not at receipt-write time.

## Production checklist

- Wrap the training step in `goodput.span("step")`, checkpoint saves in `goodput.span("checkpoint")`, eval in `goodput.span("eval")`, compile warmup in `goodput.span("compilation")`, and the dataloader pull in `goodput.span("data_loading")`.
- Write the run header from `report.py` (or its production equivalent) at job start and include git provenance, GPU info, system info, and cost info.
- Tag every receipt artifact with an explicit schema version; do not silently add fields.
- Keep `MetricsPusher` push interval >= 15 s and respect the per-time-series rate limit.
- Keep on-demand profiling behind `SIGUSR1` / `SIGUSR2` and never enable it by default on a long run.
- Reference profiler-trace artifacts by URI in the receipt; do not inline them.
- Validate every ablation receipt with its schema's `validate()` before writing.
- Run the dashboard against the receipt store, not against raw logs.
- Treat any change that moves step time as an observability cost regression and bisect it before merging.
- Persist receipts to durable storage co-located with the checkpoint, so a recovered checkpoint always has its receipt.

## References

- `goodput.py`, `report.py`, `temporal_perf.py`, `modal_receipts.py`, `receipt_schema.py`, `observability.py`
- CHANGELOG entries describing the run-receipt schema, the structured-step log format the parser depends on, and the Cloud Monitoring / OpenTelemetry wiring.
- [MaxText — Google JAX-based LLM trainer with GoodputRecorder/GoodputMonitor patterns]
- [OpenTelemetry specification — opentelemetry.io]
- [Google Cloud Monitoring API rate limits — cloud.google.com/monitoring/quotas]
- [The Tail at Scale — Dean and Barroso, CACM 2013]
