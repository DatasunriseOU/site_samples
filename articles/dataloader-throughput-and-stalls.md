---
title: "Dataloader throughput and stalls: making the input pipeline a first-class perf concern"
description: "Packed-rows schema, prefetch depth, IO budget per step, and the host-side bottlenecks we hit at 64K context — plus the XLA-friendly path that makes the input pipeline boring again."
date: "2026-04-18"
tags: ["dataloader", "throughput", "xla", "performance", "data"]
---

The fastest way to ruin a perfectly good optimizer is to starve it. This post is the engineering record of why we treat the input pipeline as a first-class performance concern, what the packed-rows schema actually looks like in MegaCpp, the host-side bottlenecks long-context training exposes, and the loader choices that keep step time stable instead of erratic.

## Why MegaCpp cares about this

Long-context pretraining leaves very little room for waste in the input pipeline. At 64K context with enriched metadata such as structure ids, chunk boundaries, call/type edges, AST node types, edit ops per token, and hunk ids per token, naive Arrow-to-numpy-to-torch decoding can dominate host-side prep. On TPU, the wrong shape decision in the loader does not just slow a step; it can trigger an XLA recompile.

The other reason this matters is honesty. If the input pipeline introduces variance, every benchmark we publish becomes noisy. Per-run receipts have to be comparable across weeks of code churn, which means the loader's own contribution to step time has to be both small and stable.

## Public loader contract

The MegaCpp loader is built around one simple rule: every training step should see the same batch shape. On disk, that means fixed-length packed parquet rows. Offline preparation tokenizes documents, aligns packs on BOS boundaries, runs a best-fit packing pass, and writes a uniform row schema that the runtime can stream without shape surprises.

Each parquet row carries the core tensors the trainer always needs: `pack_id`, `input_ids`, `target_ids`, `loss_mask`, `doc_ids`, `valid_token_count`, and `num_docs`. Document boundaries stay recoverable from `doc_ids` without building a dense `[T, T]` mask, and the per-row counters give cheap telemetry without scanning variable-length payloads. Optional structure columns piggyback on the same fixed row, so syntax- and history-aware features still arrive in static shapes.

That fixed-length contract is the load-bearing performance decision. It keeps XLA from recompiling between batches, lets staging buffers stay resident across iterations, turns each host-to-device transfer into one contiguous copy per tensor, and preserves document boundaries through `doc_ids`-driven varlen attention instead of explicit dense masking.

The prefetch layer is intentionally small. `PrefetchIterator` is a background thread plus `queue.Queue(maxsize=1)`. The queue depth stays at one on purpose: enough overlap to hide CPU-side preparation behind the current step, but not enough to let the producer sprint ahead and chew through host memory. Exceptions cross the queue with their traceback intact, so a broken shard fails loudly instead of freezing the iterator.

The 64K-context cost story shows up in two places. First, the per-token character-length lookup that enriched data needs should be table-driven, not a per-token tokenizer decode loop. Second, FIM (fill-in-middle) augmentations and their structured variants must stay on the training-only path so inference and smoke lanes do not pay for code they do not exercise.

On the IO side, the runtime only depends on the packed parquet contract, not on how the upstream corpus was materialized. Local and remote shard discovery, caching, and recovery can evolve independently as long as the output rows preserve the same fixed schema.

The token-enrichment pass also stays offline. It maps semantic chunk kinds such as `func_signature`, `func_body`, `class_decl`, and `comment` into the integer columns the runtime consumes, and it resolves the char-to-token alignment once instead of rebuilding it per batch. After fixed-length packing, that offline alignment pass is the second-largest dataloader speedup we kept.

## How it lands in MegaCpp

The packed-rows contract lifts directly into the full training runtime. MegaCpp uses the same schema, the same fixed-length invariant, the same `doc_ids`-driven varlen attention semantics, and the same offline-enrichment discipline. What changes on the way in is mostly cleanup.

Dropped: lazy experimental imports once the production lane only shipped the FIM variants it committed to support. Dropped: the runtime corrupt-shard recovery path, because production runs require pre-staged shards on local NVMe and a missing or bad shard is a hard failure rather than a runtime fixup. Dropped: the legacy flat-text producer surface; the production loader only reads packed-rows parquet emitted by the offline packer.

Rewritten: the prefetch wrapper. MegaCpp's lightweight Python queue is enough for a reference stack, but the production loader uses an Arrow-backed pipelined reader with explicit IO budgets and a memory-mapped row-group cache so host-side prep stays bounded per step. The semantics are the same: one batch ahead, natural backpressure, and real exceptions when shards fail.

Lifted to a kernel/Pallas-friendly path on TPU: the per-token char-length mapping. On TPU v6e we move the char-to-token bisect into a precomputed integer tensor we shard with the data axis, so even the structure-aware enriched columns end up as static-shape gathers rather than CPU-side numpy loops.

The XLA-friendly invariants are non-negotiable on the production TPU lane. No data-dependent shapes anywhere in the dataloader output. No `.item()` calls, anywhere, ever, on the data path. No dense attention mask construction. No per-batch tokenizer calls. The loader either produces a static-shape batch or it does not produce a batch.

## Choices and what we kept

The decisions that matter here cluster around three themes.

First, packing telemetry. The `--packing_telemetry` flag wires `PackedDocTelemetry` so utilization can be measured (`valid tokens / total slots`), alongside `docs_per_row`, `avg_doc_tokens`, and `cropped_doc_frac` across packing policies such as `packed_rows`, `best_fit`, and `single_doc`. It is opt-in, logs summaries on a configurable cadence, and stays on the host side. We kept best-fit because telemetry is useful only when it informs a stable policy choice rather than churning the packer every week.

Second, host-device sync elimination. This is not strictly a dataloader-only topic, but it belongs to the same discipline: anything in the hot path that forces a host-device sync defeats the loader's overlap budget. Regressions of that class are blockers.

Third, the dataloader-packed-rows test surface. Packed-row schema tests, metadata tests, and dataloader contract tests need to pin the loader's behavior together. The key point is the coverage shape, not one internal counter or one-off benchmark number.

What we tried and dropped:

- A loader path that emitted variable-length packs and let the model handle padding. It worked on CUDA, was a recompile pit on XLA, and we abandoned it.
- A path that built `attention_mask` matrices in the loader for safety. It doubled the H-to-D transfer cost at 64K context and recovered no correctness because varlen FA via `doc_ids` already enforces the boundaries.
- Bigger prefetch queues. `maxsize=2` and above add memory pressure quickly; `maxsize=1` is the safe default unless a trace says otherwise.
- Per-batch tokenizer calls for char-length lookup. Replaced by a precomputed table built at loader init time.

What we kept:

- Fixed-length packed rows as the on-disk contract.
- BOS-aligned best-fit packing as the default policy.
- `doc_ids`-driven varlen attention with no dense mask.
- A single-batch-ahead background-thread prefetch with `Queue(maxsize=1)`.
- Offline enrichment in the offline tokenized-enrichment step rather than per-batch.
- Opt-in `--packing_telemetry` with summary every 100 batches.

## Production checklist

- Validate every shard against the packed-row schema at ingest; reject anything missing the seven required columns or violating the fixed-length invariant.
- Treat shape variance across batches as a CI failure on the TPU lane.
- Keep prefetch depth at 1 unless a profiler trace says otherwise.
- Stage all shards on local NVMe before training; production runs do not do runtime shard recovery.
- Audit any new dataloader edit for `.item()`, `.nonzero()`, and dense-mask construction; all three are blockers.
- Run dataloader packed-row tests plus the schema contract tests on every PR that touches the loader or the schema.
- Enable `--packing_telemetry` for the first 1000 steps of any new preset rollout, then disable it for the steady-state run.
- Pin the offline enrichment recipe alongside the model checkpoint so loader semantics are reproducible.
- Record dataset shard list, schema version, and packer commit in the run receipt.
- Treat the producer story as transitional and the consumer-side schema as authoritative.

## References

- [Data preparation notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md)
- [TPU bringup notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/tpu-bringup-notes.md)
- [Tokenized enriched pipeline on TPU](https://megacpp.com/blog/tokenized-enriched-pipeline-on-tpu.md)
- [Structure embeddings and relation bias](https://megacpp.com/blog/structure-embeddings-and-relation-bias.md)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Sequence Packing for Transformer Pre-training](https://arxiv.org/abs/2107.02027)
- [Apache Arrow documentation](https://arrow.apache.org/docs/)
- [Apache Parquet documentation](https://parquet.apache.org/docs/)
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)
