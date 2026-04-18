---
title: "Dataloader throughput and stalls: making the input pipeline a first-class perf concern"
description: "Packed-rows schema, prefetch depth, IO budget per step, and the host-side bottlenecks we hit at 64K context — plus the XLA-friendly path that makes the input pipeline boring again."
date: "2026-04-18"
tags: ["dataloader", "throughput", "xla", "performance", "data"]
---

The fastest way to ruin a perfectly good optimizer is to starve it. On our hybrid stack, dataloader stalls used to be the second-largest contributor to bimodal step time after compile warmup. This post is the engineering record of why we treat the input pipeline as a first-class perf concern, what the packed-rows schema actually looks like in the prototype, the host-side bottlenecks we hit at 64K context length, and the changes we shipped so the loader stops being the thing that explains a weird `tok/sec` curve.

## Why MegaCpp cares about this

Pretraining steps on the dense+MoE preset run in the low single-digit seconds on H200:8 and similar order on TPU v6e once everything is warm. That puts the per-step IO and host-prep budget at well under a second if we want the loader to be invisible. At 64K context with enriched metadata — structure ids, chunk boundaries, call/type edges, AST node types, edit ops per token, hunk ids per token — the row payload is large enough that naive Arrow-to-numpy-to-torch decoding is enough to push the loader over budget on its own. And on TPU, the wrong shape decision in the loader does not just slow a step; it triggers an XLA recompile that costs minutes.

The other reason this matters is honesty. If the input pipeline introduces variance, every benchmark we publish becomes noisy. Per-run receipts have to be comparable across weeks of code churn, which means the loader's own contribution to step time has to be both small and stable.

## What we built in the POC

The prototype's loader lives in the dataloader implementation, with the on-disk contract in the packed-row schema layer, the offline materialization in the offline tokenized-enrichment step and its schema definitions, and the parquet-shard discovery and download logic in the dataset-loading layer. The shape of the contract is the most important thing in this section, because every perf decision downstream falls out of it.

Each parquet row is a fixed-length pack. The required columns are `pack_id`, `input_ids`, `target_ids`, `loss_mask`, `doc_ids`, `valid_token_count`, and `num_docs`, defined in the packed-row schema layer. The packer (the offline packed-row materialization stage) takes variable-length tokenized documents, BOS-aligns them, runs a best-fit packing pass, and emits rows of a uniform sequence length. Document boundaries inside a pack are recoverable from `doc_ids` (zero is reserved for padding doc), and `valid_token_count` plus `num_docs` give the loader O(1) telemetry per row without scanning. Optional enriched columns piggyback on the same row: `token_structure_ids`, `token_chunk_starts/ends/kinds/dep_levels`, `token_call_edges`, `token_type_edges`, `token_def_use`, `token_ast_depth`, `token_ast_node_type`, `hunk_id_per_token`, `edit_op_per_token`, plus the chronology columns (`commit_hash`, `commit_timestamp`, `parent_hashes`, `is_merge_commit`, `repo_stable_id`, `filepath_stable_id`, `file_local_commit_index`).

The fixed-length design is the single most important perf decision in the loader. It means: every batch has identical shape, so XLA never recompiles between batches; pre-allocated staging buffers can be re-used across iterations with no GC pressure; the H-to-D transfer is a single copy of a contiguous block per tensor; and document-boundary masking rides on `doc_ids` via varlen FA, so the loader never has to materialize a dense `[T, T]` attention mask. Public change notes record the second-order benefit: we never emit `attention_mask=` to `GPT.forward`, which is the construction-by-default form of Megatron's `--no-create-attention-mask-in-dataloader` mode.

`PrefetchIterator` in the dataloader implementation (around line 295) is the small piece of glue that makes the loader cooperative with the GPU/TPU step. It is a ~20-line wrapper around a background thread and a `queue.Queue(maxsize=1)`. The `maxsize=1` is deliberate: it gives natural backpressure so the producer thread does not race ahead and burn host memory while the consumer falls behind. Exceptions propagate through the queue with traceback intact so a bad shard surfaces as a real stack rather than a frozen iterator. The comment in the source says it eliminates "bimodal throughput from data stalls," and that is exactly what it was written to do — without prefetch, the H-to-D copy and the next batch's CPU-side prep serialize against the step.

The 64K-context cost story shows up in two places we had to fix. First, the per-token-character-length lookup that enriched-data needs (to map between tokens and the original character spans for chunk relations) used to do a per-token tokenizer decode loop. The comment in the source records the cost: "~150ms/batch at 64K." We replaced it with a single numpy index into a precomputed `_token_char_lengths_table` built once at dataloader init from the tokenizer, dropping per-batch cost to ~0.01ms. Second, the FIM (fill-in-middle) augmentations and their structured/IFIM/SRI variants are gated by `split == "train"` and lazy-imported from prototype-only experimental imports so the inference path and the dev lane never pay for code they do not exercise.

On the IO side, the dataset-loading layer deliberately keeps the producer story behind a stable consumer-side interface. Its parquet discovery helpers resolve either local or remote object-store URIs, the local cache directory is materialized at import time with a temp-directory fallback, and a recovery helper lets us re-download a corrupt shard at runtime via the configured object-store bucket variable. Producer paths can churn (we have at least three: a Rust chunker plus libclang indexer that emits enriched JSONL, a legacy flat-text producer, and a binary token dataset path that historical jobs used) without forcing loader changes — the loader only sees parquet that satisfies the packed-row schema.

the offline tokenized-enrichment step is the offline helper that materializes token-level enriched parquet rows from the raw enriched JSONL produced by the indexer. It owns the mapping from semantic chunk kinds (`other`, `preamble`, `func_signature`, `func_body`, `class_decl`, `class_member`, `comment`, `typedef`, `namespace`) to the integer ids the runtime sees, and the inverse char-to-token bisect that gives the per-token semantic columns. Doing this work offline rather than per-batch is the second-largest source of dataloader speedup after fixed-length packing.

## How it lands in MegaCpp

The packed-rows contract lifts as-is into the production trainer. The MegaCpp production codebase consumes the same schema, with the same fixed-length invariant, the same `doc_ids`-driven varlen attention semantics, and the same offline-enrichment discipline. What changes on the way in is mostly cleanup.

Dropped: the lazy prototype-only experimental imports imports, because the production lane only ships the FIM variants we have committed to and the import hooks become dead code. Dropped: the runtime corrupt-shard recovery path, because production runs require pre-staged shards on local NVMe and a missing or bad shard is a hard CI failure rather than a runtime fixup. Dropped: the legacy flat-text producer surface; the production loader only reads packed-rows parquet emitted by the offline packer.

Rewritten: the prefetch wrapper. The prototype's `PrefetchIterator` is a single-thread Python queue. The production loader uses an Arrow-backed pipelined reader with explicit per-iteration IO budget and a memory-mapped row-group cache, so the host-side prep cost stays bounded per step. The semantics are identical (one batch ahead, natural backpressure, exceptions propagate with traceback), but the implementation gets to use the production runtime's IO scheduler.

Lifted to a kernel/Pallas-friendly path on TPU: the per-token char-length mapping. On TPU v6e we move the char-to-token bisect into a precomputed integer tensor we shard with the data axis, so even the structure-aware enriched columns end up as static-shape gathers rather than CPU-side numpy loops.

The XLA-friendly invariants are non-negotiable on the production TPU lane. No data-dependent shapes anywhere in the dataloader output. No `.item()` calls, anywhere, ever, on the data path. No dense attention mask construction. No per-batch tokenizer calls. The loader either produces a static-shape batch or it does not produce a batch.

## Ablations and what we kept

The CHANGELOG entries that matter here cluster around three themes.

First, packing telemetry. The `--packing_telemetry` flag wires `PackedDocTelemetry` so utilization can be measured (valid tokens / total slots), `docs_per_row`, `avg_doc_tokens`, and `cropped_doc_frac` across all packing policies (`packed_rows`, `best_fit`, `single_doc`). It is opt-in, logs a summary every 100 batches via an env-controlled cadence knob, and has zero device overhead because all state is plain Python. The takeaway: BOS-aligned best-fit packing on our corpus reaches ~100% slot utilization with ~35% of tokens cropped at T=2048; longer T has smaller crop fractions but more CPU-side packing cost. We kept best-fit; we did not switch to a fancier packer because the telemetry showed the simpler one was already on the right side of the diminishing-returns curve.

Second, host-device sync elimination. Changelog entries around two hot-path runtime components (the `_safe_item(..., fallback=True)` family of fixes) are not strictly dataloader changes, but they belong to the same discipline: anything in the hot path that forces a host-device sync defeats the loader's overlap budget. We treat any such regression as a blocker now.

Third, the dataloader-packed-rows test surface. sanitized dataloader packed-row tests, sanitized packed-row schema tests, sanitized packed-row metadata tests, and sanitized H200 sparse validation smoke tests together pin the loader's behavior. The CHANGELOG records 45-passing runs on the H200 lane after each contract change, which is the gate we use before shipping a loader edit.

What we tried and dropped:

- A loader path that emitted variable-length packs and let the model handle padding. It worked on CUDA, was a recompile pit on XLA, and we abandoned it.
- A path that built `attention_mask` matrices in the loader for safety. It doubled the H-to-D transfer cost at 64K context and recovered no correctness because varlen FA via `doc_ids` already enforces the boundaries.
- Bigger prefetch queues. `maxsize=2` and above bought us nothing measurable on H200:8 and added memory pressure; `maxsize=1` is the right default.
- Per-batch tokenizer calls for char-length lookup. Cost ~150ms/batch at 64K; replaced by the precomputed numpy table at init time.

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
- Run sanitized dataloader packed-row tests plus the schema contract tests on every PR that touches the loader or the schema.
- Enable `--packing_telemetry` for the first 1000 steps of any new preset rollout, then disable it for the steady-state run.
- Pin the offline enrichment recipe alongside the model checkpoint so loader semantics are reproducible.
- Record dataset shard list, schema version, and packer commit in the run receipt.
- Treat the producer story as transitional and the consumer-side schema as authoritative.

## References

- the dataloader implementation, the dataset-loading layer, the offline tokenized-enrichment step, the tokenized-enriched schema layer, the packed-row schema layer, the fill-in-the-middle packing path, the structure-embedding layer
- changelog entries: `--packing_telemetry`, `_safe_item` host-device sync fixes in two hot-path runtime components, dataloader-packed-rows test runs (45 passing).
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al., NeurIPS 2022]
- [Sequence Packing for Transformer Pretraining — Krell et al., arXiv:2107.02027]
- [Apache Arrow / Parquet columnar format documentation — apache.org]
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs — Xu et al., arXiv:2105.04663]
