---
title: "Loss Curves and the Divergence Playbook: How We Catch It at Epoch 0"
description: "The divergence playbook used on every training start: early-training spikes, NaN bisect, LR warmup shape, data-order suspects, and the monitors that catch it before step 100."
date: "2026-04-18"
tags: ["training", "divergence", "loss-curves", "monitoring", "debugging"]
---

Most training runs do not fail at the model. They fail at step 1, or step 50, or step 24,850, and someone has to figure out why. Repeated bring-up work has produced enough specific failure modes that a fixed playbook is now used against the first hundred steps of every new configuration. This post is that playbook: what spikes look like, how we bisect a NaN to a single rank or kernel, what LR warmup shape we landed on, the data-order suspects we keep checking, and the monitors that catch each of these before they become an overnight loss.

## Why this matters

An overnight run that diverges at hour six wastes more than the hour-six dollars. It wastes the next morning's investigation, invites a sloppy "we think it was the LR" postmortem, and quietly raises the bar for the next bug because nobody wants to reopen the same wound. A 100-step smoke that catches the problem before a long run starts is two orders of magnitude cheaper, and the playbook here exists to make that smoke load-bearing.

The other reason: a modern training stack can mix Muon plus AdamW, FP8 islands, MoE routing, Mamba SSM kernels, and intra-document masking. Each subsystem has its own divergence signature, and trying to debug them in aggregate is hopeless. The bisect order in this playbook reflects how often each subsystem has actually been the culprit in practice, not how interesting it is in the abstract.

## 1. What divergence looks like in real run data

Three patterns dominate.

### Immediate NaN at step 1 with a finite step 0

Almost always an optimizer or precision issue, not a model issue. The canonical case from a TPU v6e case study: step 0 loss `14.79` on a v6e-32 slice and `11.46` on v6e-16, then every subsequent step `NaN`. With `--no_muon` (AdamW only) the loss stayed finite (`14.79 -> 14.70 -> 25.9 -> 33.6 -> 39.2`). The bisect pointed at Muon's Polar Express Newton-Schulz iteration running in BF16. On CUDA the TensorCores accumulate in FP32 internally; on XLA they do not, and the Polar Express coefficients are large enough (`a = 8.15`, `b = -22.48`, `c = 15.88` on the first iteration) that five iterations in BF16 compound into a NaN. The fix is one line: detect `PJRT_DEVICE=TPU` and run the iteration in FP32. CUDA stays BF16. Both TPU pods recovered.

### Early gradient spike that does not NaN

H200 4-step receipts on a representative Hopper recipe consistently show a `gnorm 1160 -> 3648 -> 7200` pattern across the first three steps under one topology; an alternate `activation_memory_budget=1.0` topology showed `1040 -> 4800 -> 7392`; a `dsa_query_chunk_size=16` lane showed `1072 -> 3904 -> 6976`. Same severe pattern under otherwise different knobs. The spikes do not break training (clip catches them, the EMA settles by step 50), but they are a real signal: the model is taking large early steps under the current init and LR schedule, and any change to LR shape or initialization should be evaluated against this baseline.

### Late-training transient

The d24 hybrid 877M run took a transient loss spike around step 24,850 (loss jumped from ~0.8 to ~3.4) and fully recovered by step ~25,700 (back to ~0.85). The failure mode here is not the spike, which Muon + AdamW handled, but that the step 25K checkpoint was saved during it. Eval on that checkpoint dropped from 11.0% compile (step 20K) to 3.1% (step 25K). The lesson is operational, not algorithmic: do not save checkpoints during a `gnorm` excursion.

## 2. The NaN bisect

When step N produces a non-finite loss or gradient, we run a deterministic bisect. The order matters because it cuts the search space the fastest.

1. **Disable Muon.** If the run goes finite, the issue is in Muon or its interaction with the precision policy. This caught the TPU Polar Express NaN above.
2. **Force BF16 across all paths** (no FP16, no FP8). If the run goes finite, the issue is in a low-precision island. The most common case is FP8 e4m3 saturating on the first backward of a freshly-initialized model; the optimizer's finite-check-and-skip prevents weight pollution and subsequent steps stabilize as the FP8 amax history fills out. Making the strict `check_for_nan_in_grad=True` guard opt-in lets a run pass through the FP8 warmup transient without masking the root cause.
3. **Bisect by rank.** The pre-collective sanitization step `nan_to_num`s the flattened gradient before the reduce-scatter; we temporarily disable it (`MEGACPP_SKIP_PRE_REDUCE_NAN_CHECK=1`) and log per-rank `is_finite(grad).all()` before the collective. A single rank producing a NaN points at MoE routing overflow, a Mamba convolution edge case, or an experimental kernel firing on that rank's data.
4. **Bisect by block.** Disable EBlocks (`--no_moe`), then RBlocks, then MBlocks, in that order. The order reflects historical likelihood: MoE has been the most common source of late-training spikes; M2RNN is rarely the cause; Mamba kernels are occasional contributors at specific seq_lens.
5. **Re-enable with the simplest config that reproduces.** The reproducer is the asset; the fix is downstream.

The bisect typically finishes in under an hour because every step is a four-step smoke (`--save_every=0 --core_metric_every=0 --sample_every=0 --eval_every=0 --warmup_ratio=0.0`), the same shape we use for the H200 modal matrix receipts.

## 3. LR warmup shape

We tried several warmup shapes; the production default is linear warmup followed by Megatron-style full-range cosine decay to a `final_lr_frac` floor. The implementation in the main training entrypoint supports `linear`, `cosine`, `WSD`, `inverse-square-root`, and a few exotic options (`exponential`, `minus_sqrt`); we ship `cosine` because it is what our reference Nemotron-class stack uses and because it produced the cleanest curves on our hybrid configs.

The warmup ratio matters more than the shape. We default to `warmup_ratio=0.03` (so 3% of `num_iterations` is warmup); the formula is `warmup_iters = round(warmup_ratio * num_iterations)`, the multiplier during warmup is `(it + 1) / max(1, warmup_iters)`, and after warmup the multiplier is the cosine decay to `final_lr_frac`.

| Shape | When we use it | Notes |
| --- | --- | --- |
| linear-warmup + cosine | default training | cleanest curves on hybrids |
| linear-warmup + WSD | long runs with planned mid-rate plateau | works, slightly noisier final |
| linear only | smoke and 4-step receipts | warmup off, deterministic |
| inverse-sqrt | parked | does not beat cosine on our data |

### Initialization changes are warmup changes

An init change that is meant to be cosmetic — for example, zeroing the outgoing weights of "noop" experts — can act like a discontinuous schedule. In a 24-layer transformer with residual connections, zeroing outgoing weights creates a sudden void in the residual stream and spikes the loss by 0.01-0.05 with ~200 steps of recovery. We weaken to `std * 0.1` instead, which preserves signal flow. Anything that discontinuously changes the residual stream is a divergence risk, and warmup will not save you from it.

## 4. Data-order suspects

Every "the loss spiked at step X" investigation eventually checks the data shard at step X. We log the current shard name every 100 steps specifically for this. The suspects we check, in order:

1. **A single bad shard.** If the spike correlates with a particular parquet file across reruns, the file is the answer. Our pipeline shuffles within shards (`random.Random(42).shuffle(texts)` at ingest), so a bad document is bounded to one shard rather than smeared across many.
2. **A shard-boundary effect.** Our packed-doc pipeline can produce an unusually long contiguous run of similar-domain text at a shard boundary, behaving like a tiny domain shift. Shows up as a correlated `gnorm` and `loss` bump that resolves in <100 steps.
3. **A curriculum transition.** Some training corpora have explicit curriculum stages; transitions are obvious in the curve and are expected. We mark them in the run log so they are not mistaken for instability.
4. **Token-distribution drift.** We periodically compare the rolling token-frequency histogram to the pretraining baseline. A KL spike here usually points at a tokenizer edge case rather than a data issue.
5. **Determinism check.** Same seed, same shard order, same loss within numerical tolerance for the first ~20 steps. If determinism breaks, the issue is in the loader or the FSDP2 sharding, not the data.

The shuffle-by-rank pattern matters too. We seed per-rank with `(global_seed, rank, epoch)` so ranks see different data within a microbatch boundary; this is the same convention as Megatron's data sampler. Getting this wrong is silent: the model trains, but it trains on the same tokens N times per step.

## 5. The monitors that catch it at epoch 0

We run four monitors on every step. They are cheap; we keep them on in production.

### Grad-norm EMA detector

Lives in the main training entrypoint. Maintains `_gnorm_ema` with `alpha = 0.01`, compares each step's `grad_norm` against the OLD EMA before updating (so the spike does not inflate the EMA and clip the ratio at `1/alpha = 100`), and prints `[GRAD SPIKE WARNING]` at >10x EMA and `[GRAD SPIKE CRITICAL]` at >100x. The ratio-cap detail matters: an earlier version updated the EMA first and then compared, which made CRITICAL effectively unreachable.

### Non-finite optimizer guard

Detects non-finite gradients before sanitization. With `--skip_nan_steps`, the optimizer step is skipped entirely (Megatron pattern), `_step_skipped = True` is set on the wrapper, and the LR scheduler does not advance. We log the skip; a steady drip of skipped steps is a sign that the precision policy is wrong, not that the data is bad.

### Loss-curve 2x rule

Rolling 2x-of-average rule from the design doc: if the smoothed loss exceeds 2x the recent average, save a checkpoint and pause for investigation. The d24 hybrid spike at step 24,850 should have triggered this; it did not because the alert pre-dated the EMA implementation. We added it to the loop after that incident.

### Throughput leading indicator

A sudden drop in `tok/sec` at step N often precedes a loss event by a few hundred steps; the cause is usually graph recompilation or a comm-pattern change, but occasionally it is the optimizer entering a region where per-step work changes. We use it as a leading indicator, not a diagnostic.

The reporting line is one we look at on every step:

```text
step 00050/30000 (0.17%) | loss: 4.2371 | lrm: 0.55 | dt: 412.10ms |
  tok/sec: 254,128 | mfu: 17.8 | gnorm: 1.4231 | total time: 0.34m
```

`lrm` is the LR multiplier from the schedule, `dt` is the step time, `gnorm` is the post-clip grad norm. Reading this line, the warmup ramp, the early-step compile tax, and the gnorm settling are all visible in the first thirty steps. A new config that diverges in the first hundred steps shows it here, in a single line per step, before any eval ever runs.

## 6. What we throw out at step 100

If the playbook says diverged, we kill the run. We keep one log artifact (the last 200 steps of training output, the optimizer state diagnostic, the per-rank `is_finite` snapshot), tag the directory, and start the next candidate. The cost of a failed 100-step smoke is small; the cost of an overnight run that diverges at hour six is large; the cost of a checkpoint saved during a spike is larger still.

## What we kept and what we threw away

Kept: the bisect order (Muon -> precision -> rank -> block), the linear-warmup-plus-cosine default with `warmup_ratio=0.03`, the four step-level monitors, per-100-step shard logging, the four-step smoke shape, and the "do not save during a `gnorm` excursion" rule. The single-line training log with `loss / lrm / dt / tok/sec / mfu / gnorm` stays as the universal diagnostic surface.

Threw away: zero-init for noop experts (replaced with `std * 0.1`), the inverse-sqrt and exponential LR shapes (do not beat cosine on our data), strict-NaN-grad as a default (made it opt-in to survive the FP8 warmup transient), and EMA-after-compare (made CRITICAL unreachable). The 5-shape LR library is also slated to collapse to two shapes (linear-warmup-cosine and linear-warmup-WSD) at the next refactor.

## What still hurts

Three honest gaps. The grad-norm EMA monitor is a heuristic; a real anomaly detector with a per-shard prior would catch the d24-style mid-training spike earlier. The data-order bisect is manual: we have shard logs and the seed manifest, but reproducing a single problematic batch from a multi-rank run is still a half-day exercise, and a "replay this exact step" tool is overdue. And the LR schedule library is more optionality than we use.

## References

- [Data preparation notes](../docs/data-prep-notes.md)
- [Muon optimizer repository](https://github.com/KellerJordan/modded-nanogpt)
- [PyTorch distributed documentation](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/)
