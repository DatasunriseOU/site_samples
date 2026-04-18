---
title: "Attention sinks and sink/spike telemetry on the TPU v6e long-context path"
description: "How we measure attention sink ratios and activation spikes inside an XLA SPMD training step on TPU v6e without paying for a host-device sync on every layer, and what the dashboards actually tell us."
date: "2026-04-18"
tags: ["tpu", "v6e", "xla", "spmd", "attention-sinks", "telemetry"]
---

The cross-path post on long context is the architectural story: YaRN, RNoPE, packed-document masking, gated attention. This one is narrower. It is about the telemetry pipeline that decides which of those mitigations actually moved the numbers, and specifically about how that pipeline runs on a TPU v6e SPMD training step without crashing the compiler, melting the dashboards, or ten-minute-recompiling between logging steps. The interesting parts are not the metrics themselves; they are the constraints XLA imposes on collecting them.

## Why MegaCpp cares about this

The C++ specialists train past 16K tokens with packed documents, gated attention, and a hybrid of dense and sparse attention layers. Two unrelated pathologies bite at long range. The first is the **attention sink**: a small set of tokens, often the first token of each row or each document, soak up a disproportionate fraction of softmax mass regardless of content. The second is **activation spikes**: a few hidden units in a few tokens grow to magnitudes orders of magnitude larger than their neighbors. The two correlate, but only loosely. A model can post a healthy max-activation number while leaking attention into position 0; another can have well-behaved attention while the residual stream blows up at layer 47. We have to track both, separately, on every long-context experiment.

On H200 the cost of that tracking is negligible because the runtime is eager. On TPU v6e it is not. Anything that calls `.item()` inside the forward pass triggers a host-device sync under XLA lazy execution. A naive sink tracker would do that on every layer of every microbatch, which on a depth-52 hybrid preset means dozens of syncs per step, each one stalling the pipeline. So the constraint is concrete: measure two phenomena, in two telemetry streams, on every dense and sparse attention layer, with a per-step cost that is measurable in single-digit percent of step time and a per-non-logging-step cost of zero.

## What we built in the POC

Three modules in the experimentation codebase carry the work: the public sink-telemetry sample, the public outlier-telemetry sample, and a thin compositional layer in the public spike-telemetry sample. The split is deliberate, and the docstring on the public sink-telemetry sample quotes our own RFC at length on why: sink reduction is not the same as spike reduction. Mixing them in one tracker pretends the two metrics share a denominator. They do not.

the public sink-telemetry sample defines `SinkTelemetry` (alias `SinkMetrics`). It runs as a side observer inside `CausalSelfAttention.forward`, after Q/K projection and QK-norm and before the flash attention call. The `track_attention_weights` entry point detaches Q and K immediately so it never participates in the backward graph, expands GQA key heads to match query heads via `repeat_interleave`, and computes the sink ratio as the softmax weight on either global position 0 or the first token of each query's document, depending on whether `doc_ids` describes packed structure.

The "softmax weight without materializing the full T-by-T matrix" trick is the load-bearing piece. `_compute_global_sink_ratio` computes `score_0 = (q_probe * k_0).sum(-1) * scale` directly, takes a `logsumexp` over a probe of query positions and a subsample of keys (capped at `_MAX_PROBE_QUERIES = 128` and `_MAX_KEYS = 2048`), and returns `exp(score_0 - lse)` as the sink weight. For document-local tracking, `_compute_doc_local_sink_ratio` walks each batch element, finds each document's first-token position, and applies a causal-and-same-doc mask before the same logsumexp trick. The sampling cost is bounded by construction; the kernel inside is a pair of small einsums that XLA happily fuses into the surrounding attention graph.

The matching module on the activation side is the public outlier-telemetry sample, which exposes `OutlierTelemetry` (alias `SpikeMetrics`). It tracks three things per layer: `max_activation`, `outlier_fraction` (fraction of activations beyond 6 sigma), and excess `kurtosis`. Collection is via a forward hook registered on `model.transformer.h`. The hook is exactly the place where the TPU constraint shows up. The internal `_compute_stats` calls `flat.abs().max().item()`, `var.item()`, and one or two more scalar reads to decide whether to compute outliers and kurtosis at all. Each `.item()` is a host-device sync under lazy XLA. The fix is the `enabled` flag: when `False`, the hook is a literal no-op. The training loop in the TPU training launcher arms it only on logging steps (`_outlier_telemetry.enabled = step % 100 == 0`). Off-step cost is zero, on-step cost is bounded.

the public spike-telemetry sample adds the composition layer. `SinkSpikeSeparator` takes the per-layer sink summary from one tracker and the per-layer spike summary from the other and assigns each layer to one of `CLEAN`, `SINK_ONLY`, `SPIKE_ONLY`, or `BOTH`. The thresholds are configurable (`sink_threshold=0.3`, `magnitude_threshold=50.0`, `kurtosis_threshold=10.0`, `outlier_threshold=0.001` are the defaults), and the output is a structured dict that the dashboard layer can render directly. This is what makes the RFC's "sink reduction is not spike reduction" line operational: a mitigation that flips `SINK_ONLY` layers to `CLEAN` without touching `SPIKE_ONLY` layers is doing exactly what we asked, and we can see that on the per-layer counts before and after.

The training-loop integration is the other half. In the SPMD path of the TPU training launcher, the model exposes the sink collector as `orig_model._sink_telemetry_collector` (the GPT module creates it on the meta-init pass when `config.sink_telemetry` is true). On the logging step the loop pulls `get_sink_summary()` out, splits the per-layer dict into `dense` and `sparse` aggregates so dense `CausalSelfAttention` layers and DSA/clustered-sparse layers get separate dashboard lines, and emits both the mean/max ratios and the per-layer breakdown to the run log. The outlier loop does the same: pull `get_summary()`, log `max_activation`, `mean_outlier_fraction`, `max_outlier_fraction`, `mean_kurtosis`, `max_kurtosis`, plus the per-layer max-activation and kurtosis, then call `reset()`. None of this is in the optimizer path; it is post-loss, post-backward, and gated on `step % log_interval == 0`.

The dashboards consume both streams as time series with a layer-index facet. The signal we care about most on the long-context runs is not the absolute magnitudes (those are noisy across configs) but the *shape* of the curve across layers and across logging steps. A run where mid-stack layers slowly accumulate sink mass over training is a different failure than one where one specific layer spikes once at step 17K, and the per-layer-per-step view is what makes the difference legible. The CHANGELOG entry from a depth-52 TPU run that hit a catastrophic loss spike at step 17115 is a good example of why: by the time the spike showed up in loss, the per-layer telemetry had already been logging the early outlier-fraction climb for several hundred steps in the affected layers.

## How it lands in deployment

The contract that lifts cleanly is the API, not the internals. `SinkTelemetry.track_attention_weights`, `OutlierTelemetry.attach`/`enabled`/`get_summary`, and `SinkSpikeSeparator.classify_anomalies` are the surface the deployed code wires against. The attention modules carry the call sites; the optimizer loop carries the gating; the dashboard schema is unchanged.

What we are rewriting on the way in:

1. The Python loop inside `_compute_doc_local_sink_ratio` iterates per batch element to compute `total_valid` and per-batch sink weights. That is fine when the probe count is small but it is the part most likely to surprise XLA. The deployment version replaces the per-batch loop with a vectorized scatter that XLA can trace as a single graph, removing the variable-length control flow that currently forces the document-local path through a slower fallback.
2. The `.item()` calls inside `_compute_stats` are batched into a single `torch.stack(...).cpu()` on the logging step. XLA still pays one sync, but it pays it once per logging step instead of three to five times per layer.
3. The collector lifecycle is moved from "attribute on the model" to a typed context manager, so the `enabled` flag and the sampling caps live in one place and the training loop does not have to know whether the collector exists.

Dropped: the per-layer `print0(...)` line emitted every logging step. The dashboards subsume it and the line costs another host print on what is supposed to be a fast loop.

Moved to a kernel path: nothing. The whole point of this telemetry is that it stays out of the optimized attention kernel; making it a kernel would be moving the wrong direction.

Becomes a feature flag: the on/off toggle and the logging interval. The collectors are constructed unconditionally in the training entry point so the schema is stable, and the flag controls whether the observer actually runs and whether the summary is emitted. This is also what avoids the recompile risk: making the flag a config field rather than a structural toggle keeps the XLA graph identical between runs.

## Ablations and what we kept

The CHANGELOG history on this path is an honest record of what XLA punished us for. A few highlights worth carrying forward.

The sampling caps survived contact. `_MAX_PROBE_QUERIES = 128` and `_MAX_KEYS = 2048` were set after a sweep that compared the subsampled sink ratio against a full-T computation on the dense path; the subsampled estimate was within sampling noise of the full computation across every layer we measured, and the wall-clock cost dropped from "noticeable on a logging step" to "lost in the noise." Higher caps did not move the metric; lower caps started losing the long-tail layers.

The dense-versus-sparse split survived contact. Early versions reported a single `mean_sink_ratio` and we kept being surprised by it: a long-context run with a healthy mean would have a few sparse layers running at 0.6 sink ratio while the dense layers were at 0.1. Splitting the report into `dense_mean_sink_ratio` and `sparse_mean_sink_ratio` (the same `per_layer_type` field that `track_attention_weights` records via the `layer_type` argument) made the bimodality obvious and is the form the dashboards consume.

The `enabled` toggle survived contact, and it is the single most important line in the spike module. A run that left `enabled=True` on every step looked like it was running fine until the first synchronous all-reduce and then the step time blew up by 2x to 4x, depending on layer count. The CHANGELOG has an entry where the rng_uniform rewriting bug crashed gated-attention plus sink_telemetry on XLA; the workaround was the same gating discipline, plus an explicit detach inside the sink probe.

What we tried and did not keep: a "global sink severity" scalar that combined sink and spike into one number. It looked nice on a single dashboard tile and it was useless for diagnosis, because it conflated exactly the two phenomena `SinkSpikeSeparator` exists to separate. We deleted it and went back to two separate lines.

## Production checklist

- The sink collector and the spike collector are constructed unconditionally; their `enabled` flags decide whether they do work.
- `OutlierTelemetry.enabled` flips on logging-step boundaries only. Off-step cost must be a literal no-op (the assertion is on the hook body).
- `SinkTelemetry.track_attention_weights` is called from inside both the dense `CausalSelfAttention.forward` and the sparse attention modules, with the correct `layer_type` argument so the dashboard split is well-defined.
- `_MAX_PROBE_QUERIES` and `_MAX_KEYS` stay at 128 and 2048 unless an ablation explicitly shows the caps clipping a real signal.
- Telemetry tensors are detached at the entry point. No telemetry call ever participates in autograd.
- The summary is reduced to scalars on the host once per logging step. No `.item()` lives in the per-layer hook.
- The XLA SPMD mesh for the run is fixed before the collectors arm; the telemetry path adds zero `mark_sharding` calls and zero new sharding annotations.
- Dense and sparse aggregates ship to the dashboard separately; the per-layer-per-step view is the primary diagnostic surface.
- The `SinkSpikeSeparator` classification is computed offline from the logged summaries, not inside the training loop.
- A run that disables both collectors must produce a bit-identical loss curve to a run that enables them on a 100-step interval.

## Telemetry snapshot

| Metric | Collected where | XLA cost |
|--------|-----------------|----------|
| Sink ratio | per-layer attention scores | async transfer, no sync |
| Activation spike count | packed-doc boundary | scalar reduce, batched |
| Softmax denominator floor | FA backward | cheap reuse of existing tensor |
| KV-cache drift | long-context eval only | off critical path |

```python
# emit without synchronization point: stage metrics to a device-side ring and
# flush on a fixed cadence, not every step
sink_ratio = (attn_probs[..., 0] > sink_threshold).mean(axis=-1)
telemetry.stage("sink_ratio", sink_ratio, cadence_steps=50)
```

## References

- the public sink-telemetry sample (POC)
- the public spike-telemetry sample (POC)
- the public outlier-telemetry sample (POC)
- the main model runtime module (the attention forward call sites that invoke `track_attention_weights`)
- the TPU training launcher (the SPMD training loop, sink/outlier logging blocks)
- the public XLA flags sample (TPU generation detection and libtpu flag application)
- [Efficient Streaming Language Models with Attention Sinks - Xiao et al., ICLR 2024]
- [Massive Activations in Large Language Models - Sun et al., 2024]
- [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing - Bondarenko et al., NeurIPS 2023]
