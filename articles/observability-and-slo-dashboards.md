---
title: "Observability and the Three Dashboards We Actually Live With"
description: "Metrics, traces, and the training / infra / serving dashboard layout that keeps an eight-specialist C++ ensemble debuggable at 3am."
date: "2026-04-18"
tags: ["observability", "metrics", "traces", "slo", "dashboards", "serving"]
---

Every serving system eventually gets an observability story. Most are bad — panels accrete one at a time, each answering a question that mattered once, until forty charts fight for a single page and nobody knows which to trust. We have been through that cycle twice, once on an earlier research cluster and once on the current serving of the eight-specialist C++ ensemble. This post is what we landed on after both rounds of pruning.

## Why this matters

Observability is the part of the system that decides how long an incident takes and whether the post-mortem is grounded. Get it wrong and on-call stares at forty charts that all look slightly off. Get it right and the right owners see the right signal first, with a drill-down path that does not require asking the original author for a screenshot. The whole observability layer here reduces to three dashboards, four categories of metric, two kinds of trace, and a small set of rules about what is allowed on each surface. The rules matter more than the panels.

## 1. The three dashboards

We have training, infra, and serving. Each has one owner, one SLO story, and a hard rule against bleeding into the others.

### Training dashboard

Audience: the pretraining and post-training engineers running the weekly specialist refresh. Time horizon: the duration of one training run, hours to days. Primary signals:

- Step loss and evaluation loss per specialist, per data mix, per phase of the curriculum (the simple short-context mix, the context-graph 16K mix, the repo-graph 64K mix, the structure-aware enriched mix).
- Tokens per second per device, effective MFU, and the FLOP breakdown per component (attention, SSM, MoE experts, norms, embedding).
- Gradient norms, parameter update norms, optimizer state statistics, and the loss-weight schedule (`mtp_lambda`, `moe_aux_loss_weight`, `top_lambda`, `stp_lambda`, `gateskip_lambda`).
- FP8 amax history where FP8 is in play, NVFP4 scale statistics where NVFP4 is in play.
- MoE load-balance telemetry: per-expert token counts, aux loss, router z-loss, capacity utilization.
- Data pipeline health: shard-read lag, packing ratio, document-mask statistics, FIM-rate effective values after runtime mutation.
- Checkpoint cadence, latest successful checkpoint, staleness.

Not on the training dashboard: serving latency, infra health of the serving cluster, and cost. The training dashboard has one SLO that leaves it: the weekly specialist refresh must produce a checkpoint that clears its declared correctness threshold.

### Infra dashboard

Audience: the cluster engineers running the actual silicon. Time horizon: ongoing, with a rolling 7-day default and the ability to zoom to one minute. Primary signals:

- GPU utilization, HBM utilization, HBM ECC error rates, SM clock behavior, thermal throttling.
- NVLink/NVSwitch/PCIe bandwidth counters and link error counts.
- Node health: memory pressure, page cache behavior, network throughput per interface, NIC error counts, kernel message anomalies.
- Storage: filesystem throughput on the dataset mounts, latency on the checkpoint store, space headroom on each tier.
- Power and thermal envelope at the rack-class level (we track "rack class" and "position within a rack class", never identifiers).
- Schedule churn: which accelerator pool is holding which class of workload, eviction and preemption rates at the scheduler level, how often a replica was moved.
- Cost: GPU-hours by category of workload (training, eval, serving, research), dollars per specialist-training-run at the category level, serving cost per million tokens delivered.
- Software inventory: CUDA driver version, PyTorch version, TE version, kernel library versions per deployment class. Never individual machines, only the classes.

Not on the infra dashboard: model quality or loss, per-request serving traces, or anything identifying a specific host, rack row, or region. Categories only.

Infra SLOs are simple and boring on purpose: capacity headroom on each accelerator pool above a threshold, ECC error rate below a threshold, checkpoint-write p99 under a threshold, cost variance inside a monthly band.

### Serving dashboard

Audience: the product and serving engineers; the on-call rotation. Time horizon: now, with a rolling 24-hour trend and the ability to zoom to a single second. Primary signals, per specialist:

- Time-to-first-token (TTFT) p50/p95/p99.
- Inter-token latency (ITL) p50/p95/p99.
- Queue depth p95, admission-to-first-token p95.
- Preemption rate, preemption-depth distribution, fraction of responses carrying the `preempted_once` flag.
- Paged-KV block utilization, block-pool pressure, prefix-cache hit rate per adapter.
- Per-adapter swap rate inside the scheduler.
- Speculative-decode acceptance where enabled; rejection-driven rollback overhead.
- Router signals: primary-specialist confidence distribution, shadow dispatch rate, circuit-breaker state per specialist.
- Correctness SLO: rolling 24-hour correctness score per specialist, marked degraded when it drifts below its declared pass rate.
- Error surfaces: 4xx/5xx rate to callers, tool-call failure rate, token-stream disconnect rate.

Not on the serving dashboard: training loss, raw GPU counters from a specific machine (drill-down starts on infra), or aggregate ensemble latency as a single number — we do not publish one and the dashboard does not either.

### Dashboard split, at a glance

| Dashboard | Owner | Time horizon | Primary contract |
|---|---|---|---|
| Training | Training leads | One run (hours to days) | Weekly checkpoint passes its eval floor |
| Infra | Infrastructure owners | Rolling 7d, zoom to 1m | Pools healthy, costs in band, no identifying labels |
| Serving | Serving leads + on-call | Now + 24h, zoom to 1s | Per-specialist TTFT/ITL/correctness SLOs |

## 2. The four metric categories

Every emitted metric falls in one of four buckets. The bucket dictates which dashboard is allowed to consume it.

1. SLO metrics. The named contract numbers. Few, stable, alerting attached.
2. Component metrics. The component signals that drive an SLO — KV pool pressure, MoE load balance, scheduler queue depth.
3. Health metrics. Liveness and saturation signals — GPU utilization, NIC errors, page-cache stalls.
4. Outcome metrics. Eval scores, compilation-pass rate, end-to-end correctness on canary traffic.

SLO metrics are public. Component metrics are dashboard panels behind a drill-down. Health metrics live almost entirely on infra. Outcome metrics span training and serving and are the only category that is allowed on more than one dashboard, in a fixed strip we call the "outcome bar".

## 3. Two kinds of trace

**Request traces.** OpenTelemetry-style spans from router admission through scheduler dispatch, prefill, decode, optional tool calls, and out to the caller (with draft/verify markers when speculative decoding is on). Sampling is adaptive: every error, every preempted request, every SLO breach, plus a low base rate of healthy traffic.

**Kernel traces.** Per-step GPU traces captured through the in-tree telemetry hooks and, for deeper questions, Nsight Systems on demand. Heavyweight; off by default. The serving and training dashboards have a button that says "capture a kernel trace for the next N steps on this specialist's replica class". Captured traces land in a time-bounded store and can be attached to incident tickets.

The division matters. A request trace tells you that a specific request spent 340 ms in admission, 120 ms in prefill, and produced a first token 460 ms after arrival; it cannot tell you that the prefill was slow because an SSM kernel took an unexpected compile path. A kernel trace tells you exactly that and can be unintelligible without the request context. Keeping them separate, with cross-links, keeps each trace type legible.

## 4. The rules

A small set of rules does more work than any individual panel.

- One owner per dashboard. The owner is responsible for pruning, for adding panels, and for rejecting cross-domain panels. When we did not have this rule, every dashboard accreted an "other" section that became a third of the surface area.
- Every panel has an SLO or it has a reason. Either the panel is an SLO, a direct drill-down from one, or a component metric with a specific incident in its history. "Nice to have" is not a reason. Reviewed quarterly with teeth.
- Category granularity in the dashboard layer. Metrics are emitted with full labels; dashboards consume category-level aggregations. A GPU-utilization panel on infra shows utilization by accelerator class, not a per-host heatmap. Host-level views live in the data store, not the dashboard.
- No screenshots in tickets, only links. The link includes the time range and the specialist filter. Screenshots rot.
- No Christmas-tree correlation panels. We do not put loss curves next to serving latency next to GPU utilization to look for correlations. Cross-domain analysis goes through the outcome bar and explicit post-incident work, not a unified dashboard.
- Alert on SLO breach, not metric threshold, with a minimum duration. Dashboards display thresholds where useful; alerting fires only on the SLO. Every duplicated-threshold system eventually drifts.

## 5. What breaks and how the dashboards help

A few representative incidents, sketched at the category level, show the three-dashboard split doing its job.

### A specialist's p95 ITL jumps

The serving dashboard spikes on ITL, paged-KV block-pool pressure rises, prefix-cache hit rate falls, the rest of the ensemble does not move. Component drill-down shows adapter-swap rate climbing. The incident is a caller sending a high-variance adapter stream; the fix is a caller-side circuit breaker.

### The weekly training run plateaus

Training dashboard shows flat loss on one specialist; the MoE load-balance panel drifts toward two overloaded experts; z-loss climbs. Data-pipeline drill-down shows the FIM rate was mutated mid-run and the effective value is not what the run config claims. Fix: a data-pipeline validation guard. Infra and serving stay untouched.

### HBM ECC error rate rises on a class of accelerator

Infra fires first. Serving stays green because the scheduler is already moving workload off the degraded pool; the preemption-rate panel ticks up briefly and settles. Training shows nothing because the affected pool is not carrying a training workload this week. Infra handles it; the other teams see it only through the outcome bar.

In each case, the right dashboard fires, the right team drills down, and the other two stay out of the way. That is what the split is for.

## 6. The outcome bar

The one place the three dashboards share is the outcome bar at the top: a single thin strip with the rolling correctness pass rate per specialist, the rolling end-to-end correctness on canary traffic, and the latest published evaluation scores from the most recent checkpoints. It is the only cross-domain surface we kept, and it exists so that all three teams open their dashboard and see the same product-truth number first. If the outcome bar is green and the rest is loud, on-call calms down; if the outcome bar is red and the rest is green, on-call escalates regardless. It costs little to render and resolves a surprising number of arguments.

A minimal Prometheus surface to back the bar:

```yaml
# Correctness pass rate per specialist, rolled to 1h
- record: project:correctness_pass_rate:1h
  expr: sum by (specialist) (rate(project_eval_pass_total[1h]))
      / sum by (specialist) (rate(project_eval_total[1h]))

# Canary correctness, rolled to 15m
- record: project:canary_correctness:15m
  expr: avg by (specialist) (project_canary_pass_ratio)

# Outcome-bar SLO: alert only on the bar, never on the panels
- alert: OutcomeBarRed
  expr: project:correctness_pass_rate:1h < 0.85
  for: 30m
```

## What we kept and what we threw away

Kept: three dashboards with one owner each, four metric categories, two trace systems with cross-links, adaptive trace sampling biased toward interesting traffic, an outcome bar as the only cross-dashboard surface, category-granularity labels in the dashboard layer, SLO-gated alerting.

Threw away: a unified "everything" dashboard, correlation panels across domains, host-level heatmaps in the dashboard layer, duplicated threshold logic between dashboards and alerts, per-engineer dashboards that did not survive a 30-day review, screenshots in incident tickets.

The observability story, like the serving stack it watches, gets better the more it respects boundaries. Training, infra, and serving are three different teams asking three different questions on three different time horizons. Forcing them to share a single pane of glass was, in retrospect, the most common mistake we made and the one the current layout specifically refuses to make again.

## References

- public runtime and serving interface notes
- serving-engine implementation notes
- public planning and status notes for training and backend health
