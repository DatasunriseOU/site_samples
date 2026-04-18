---
title: "The C/C++ Data Preparation Pipeline, End to End"
description: "Every stage of the MegaCpp data prep pipeline as it actually runs: ingest, dedup, license filter, doc-mask, tokenize, packed-row shard. The bugs we hit, the perf numbers we have, the gates that catch us when we are wrong."
date: "2026-04-18"
tags: ["data", "pipeline", "c++", "operations", "tokenizer"]
---

This is the operations view of how raw C/C++ source becomes packed training rows for MegaCpp. It is the sibling post to the architecture-flavored data-pipeline-story; that one frames the design decisions, this one walks the stages a junior engineer would have to debug at an incident review. Same pipeline, different altitude. The headline: it is five stages on the user-facing surface, several coexisting producer paths underneath, and a stack of quality gates that exist because each one caught a real outage.

## Why MegaCpp cares about this

The model only ever sees what the pipeline emits. A duplicated repo doubles the training weight of someone's preferred coding style. A missed license header bakes copyleft into the weights. A broken document mask lets one file leak attention into the next, and at 64K context that is the difference between repository reasoning and confabulation. The pipeline is the gatekeeper, and its quality gates are the only thing standing between a clean training run and a model that has memorized a well-known systems library surface instead of learning to write it.

Two operational facts shape every decision below. First, our hybrid C++ tokenizer is 131,072 entries, which means token IDs no longer fit in `uint16` and the on-disk format had to switch to `uint32`. Second, the consumer-side parquet contract is canonical; the producer story is still transitional in places. We design loaders to be tolerant and producers to be replaceable.

## What we built in the POC

The user-facing entry point in production is the public data-preparation pipeline, with five numbered stages: download, tokenize, format, cache, verify. Underneath, the actual work spans a semantic chunker, dedup passes, enrichment jobs, and packing stages. Walked stage by stage:

**Stage 0 - acquisition.** Eight pinned C/C++ repositories cloned shallow at explicit refs (LLVM `llvmorg-19.1.0`, Boost `boost-1.86.0` with submodules, Linux `v6.10`, fmt `11.0.0`, googletest `v1.15.0`, abseil-cpp at tip, folly at tip, grpc `v1.67.0`). Roughly 15 GB on disk after shallow clone. A separate 142-repo catalog in 16 categories is tracked as metadata for future ingestion - the public corpus catalog notes carries the URLs, size buckets, and the awkward-source notes (SQLite on Fossil, Chromium on googlesource, Unreal needing an Epic-linked GitHub account). The split between working set and catalog is deliberate: trying to ingest 142 repos on day one means fighting infrastructure, not debugging the pipeline.

**Stage 1 - ingest and chunking.** Two coexisting producers. The current mainline is a semantic chunker that walks files and splits at function boundaries with an AST-aware budget. The legacy path is an older chunking stage that still runs on some research lanes; it splits at top-level brace boundaries and budgets by approximate token count. Both write normalized text records. Performance reference points from benchmark logs: roughly 1000 files/sec on a single machine, 4.5M input files producing about 7M output documents at the 16K target bucket, total wall time around 75 minutes on a workstation-class machine. The number to internalize is the ratio: each input file becomes 1.5 documents on average.

There is one trap in this stage that will bite anyone who skips the docs. Bucket names like `4k`, `8k`, `16k`, `64k`, `128k` are *target* buckets, not exact-token guarantees. The legacy chunker budgets by a chars-per-token heuristic, which is wrong by 5-15% under our current tokenizer. A `4k_v7` shard often contains documents that tokenize to 4400-4800 tokens. The strict producer lanes are exact-token-budgeted; the older ones are not. The loader will silently crop if you trust the bucket name as a contract.

**Stage 2 - dedup.** Two passes, two scopes. First is within-corpus dedup: SHA-256 exact first, then MinHash-LSH near-dedup with 128 permutations, Jaccard 0.7, 5-token shingles. Second is cross-source dedup: provenance-aware grouping, 112-permutation LSH (14 bands of 8 rows, the MixMinMatch parameterization), Union-Find to cluster, plus optional chunk-level dedup restricted to self-contained semantic units. The whitelist matters (`FUNC_BODY`, `CLASS_DECL`, `TYPEDEF`, `NAMESPACE`); the blacklist matters more (`OTHER`, `PREAMBLE`, `FUNC_SIGNATURE`, `CLASS_MEMBER`, `COMMENT`). Deduping an `#include` block in isolation breaks surrounding code; deduping a forward declaration silently drops something downstream expects.

Operational gotcha: the MinHash-LSH index is single-process and memory-bound. On the 27.6 M-document corpus we hit ~40 GB resident before tuning the shingle iterator to stream rather than materialize. The two-pass design is not optional - exact dedup removes 30-40% before the expensive near-dedup pass even starts.

**Stage 3 - license and quality filter.** ScanCode-style license scan per file, accepting the permissive set plus weak copyleft, with Linux GPL-2.0 tagged so downstream mixes can opt in or out. Heuristic quality filters: max 1 MB per file, max line length 1000, min size 100 B, unique-lines ratio > 30%, comment-to-code ratio < 80%, strict extension whitelist. Auto-generated markers (`// Generated by`, `DO NOT EDIT`) are cheap regex wins. An entropy check above 4.5 bits/byte catches binary-in-ASCII dumps.

PII and secret scrubbing run *before* tokenization, not after. Email addresses become the synthetic placeholder `<redacted-email>`, network addresses become `<redacted-network-address>`, high-entropy strings get replaced with `API_KEY_REDACTED`, and any user paths that survived in source comments are normalized to `<redacted-path>/`. The order matters: scrubbing after tokenization means you have to round-trip through detokenize, which is fragile, and you lose the ability to fail closed on an unredacted token leak.

**Stage 4 - doc-mask preparation.** Document masking is not a separate file format in our pipeline; it is an invariant the producer respects so the consumer can recover boundaries cheaply. Every document gets a leading BOS token. That is the contract. The training loader infers `doc_ids` at runtime via a cumulative sum over BOS positions, which is O(T) per batch and requires zero storage-format change. The reason this is a stage at all: producers that pre-pack documents into rows must guarantee that BOS-aligned best-fit packing never inserts a document without a BOS, or the inferred `doc_ids` will silently merge two documents into one. We have hit this. Fix: a producer-side assert that every packed row's BOS positions equal its `num_docs` value.

**Stage 5 - tokenize.** The tokenizer has its own writeup; pipeline-relevant facts: 131,072 entries, BOS-prepended per document, `uint32` output. The offline tokenization step emits pretokenized columns and stores per-token character spans next to IDs. The spans bridge to enrichment columns (structure IDs, dep levels, AST features) that live at character level; without spans we fail closed rather than emit unaligned metadata.

**Stage 6 - packed-row shard.** Offline packing is the enriched-row packing stage, which takes per-document tokenized rows and repacks them into fixed-length training rows without truncation: best-fit decreasing, padded on the right, emitted as `input_ids` / `target_ids` / `loss_mask` plus document boundary metadata (`pack_id`, `doc_ids`, `valid_token_count`, `num_docs`, slack tokens, source provenance). The packed-row contract is a single Arrow schema and the runtime loader reads exactly those columns. Shard size is 50,000 docs per parquet file, 1024-row row-groups for fast random access, plus a the validation parquet shard carved off as the last 1% and a a completion sentinel written when the producer is done.

**Stage 7 - format and verify.** In production, the parquet shards are converted to Megatron's `.bin`/`.idx` pair through a deterministic formatter that prefers the standard indexed-dataset builder and falls back to a raw writer when that dependency is absent. `uint32` token width is mandatory at 131K vocab. Verify is `prepare_verify`, which checks `.bin`/`.idx` existence and non-empty, parses the index, asserts `max(token_id) < vocab_size`, prints the first 64 tokens of document zero, and returns non-zero on any failure. No silent fallbacks at verify time.

## How it lands in MegaCpp

The lift is small because the contract is small. MegaCpp owns the public data-preparation pipeline plus the five numbered stage scripts plus the Megatron `.bin`/`.idx` writer. Everything below stage 2 - the semantic indexer, the tokenizer, the enrichment materializer - is dispatched into an upstream build at a pinned revision. Vendoring those pieces into MegaCpp would duplicate several thousand lines of actively maintained tokenizer and indexer code; the dependency is the smaller cost.

What is being lifted as-is: the parquet schema, the tolerant loader contract, the BOS-based doc-mask inference, the offline packer, the verify gate. What is being rewritten: the legacy flat-text producer is sunset in MegaCpp; only the strict producer with exact-token budgeting and pretokenized columns ships. What is being dropped: the `uint16` binary dataset path. What is moving to a kernel path: nothing in data prep is on the kernel critical path; the structure-aware consumer side is the place where accelerator-friendly kernels matter. What becomes a feature flag: the chunk-level dedup whitelist, because some research lanes want it off to preserve more raw context.

The old multi-environment split that historically lived in separate launch paths is collapsed in MegaCpp to a single configurable data root. As long as the public data-preparation pipeline and the launcher agree on that root, no script edits are required to move data between environments.

## Ablations and what we kept

The ablations that survived contact with real GPUs are not the headline ones. They are the boring ones.

The pretokenized-vs-char-level choice. A March 2026 loader benchmarkmark with a char-level enriched lane measured 4,172 tok/sec; switching to a pretokenized semantic lane measured 17,297 tok/sec. The 4.1x came entirely from removing the per-batch char-to-token conversion in the hot loop. We kept the pretokenized path; the char-level path lives only as the offline materialization input.

The lazy-vs-eager segment materialization choice. A regression bisect found that lazy segment materialization inside the row-pack hot loop was 2.4x slower than eager precompute-once-per-doc. The fix was a one-line gate: disable lazy materialization when both TreeFFN and relation-bias style features are enabled. Steady-state throughput recovered from 562 tok/sec back to 1,324 tok/sec, matching the reference. Partial-enriched configs still use the lazy path because for them the eager precompute is wasted work.

The conv1d-vs-Python-loop choice for Mamba document masking. A doc-mask compatibility branch was forcing CUDA into a 4-iteration Python loop for the depthwise causal conv whenever `doc_ids` was non-None - which is always, with enriched data. We measured the cross-document leakage at kernel size 4 (≤3 tokens at the boundary) and decided it was negligible compared to the cost. Reverting to plain `F.conv1d(pad(ct))` recovered roughly 22% throughput, 12,069 → 14,712 tok/sec.

The bottleneck dimension on the structure embedding path. Adding `--structure_bottleneck_dim=64` recovered another ~23% on the structure-emb-enabled runs. Kept.

The shape of MinHash-LSH itself we did not ablate; we adopted the bigcode parameterization (`numPerm=128`, `threshold=0.7`, `shingleK=5`) for within-corpus and the MixMinMatch parameterization for cross-source. Both have published evidence behind them and our role here is data engineering, not novel similarity research.

## Production checklist

- Pin all repository refs by tag, never by branch. Mirror raw clones to cold storage if absolute reproducibility matters.
- Treat bucket names (`4k`, `16k`, `64k`) as targets, not contracts. Re-measure with the current tokenizer when in doubt.
- Run exact dedup before MinHash-LSH; the cheap pass removes 30-40% before the expensive pass starts.
- Restrict chunk-level dedup to the whitelisted self-contained kinds. Never deduplicate preambles, forward declarations, or class members in isolation.
- Scrub PII and secrets before tokenization, not after.
- Every document gets a leading BOS. Producer-side assert `num_docs == count(BOS positions)` on every packed row.
- `uint32` token width at 131K vocab. `uint16` is invalid and the verify gate must catch it.
- The producer writes a a completion sentinel only after the last shard is closed. Consumers must refuse incomplete directories.
- Verify is non-zero on any failure: missing index, parse error, out-of-vocab token, broken round-trip on document zero.
- The training loader fails closed on wrong-length `doc_ids`, malformed token-structure arrays, or invalid `valid_token_count`. Optional metadata may fall back to deterministic defaults; required metadata must not.
- A pipeline-level dashboard alerts on running-pod count, never on scheduled-pod count. We learned this the hard way during a Kubernetes `ImagePullBackOff` outage that produced zero data while reporting healthy.
- Keep the `_v9`/`_v10`/`_v12` producer-revision counters separate from the `v2`-`v6` schema generations in launcher configs and in human writeups. Conflating them costs onboarding hours.

## Pipeline snapshot

| Stage | Input | Output | Gate |
|-------|-------|--------|------|
| Ingest | raw repos | normalized docs | license allow-list |
| Dedup | normalized docs | unique docs | minhash-LSH threshold |
| License filter | unique docs | permissive subset | SPDX match |
| Doc-mask | permissive subset | docs + loss mask | schema check |
| Tokenize | masked docs | token streams | vocab coverage check |
| Pack | token streams | packed shards | row-validity contract |

```text
Single-stage rerun example:
- stage: pack
- slice: core_cpp
- input: tokenized shards
- output: packed shards
- row length: 8192
```

## References

- the public curriculum-mapping notes
- the public corpus catalog notes
- the public data-preparation pipeline
- [MixMinMatch - arXiv:2512.18834]
- [The Stack: 3 TB of permissively licensed source code - BigCode]
