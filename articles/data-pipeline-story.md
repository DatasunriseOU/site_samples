---
title: "Building the C++ Training Data Pipeline: What Worked, What Broke"
description: "An honest walkthrough of how the MegaCpp training data pipeline was built — source selection, filtering, dedup, tokenization, document masking, and the quality gates that catch our own mistakes."
date: "2026-04-18"
tags: ["data", "pipeline", "c++", "tokenizer", "quality"]
---

The single most important decision we made on MegaCpp was not which optimizer to use, which sparsity ratio to pick for the MoE, or even how big to make the model. It was deciding what bytes the model would ever see. This post is the long-form version of that decision: how the C++ training data pipeline was assembled, the filters and dedup stages we trust, the producer paths we still consider transitional, and the mistakes we made on the way that are worth writing down so we do not repeat them.

## Source selection: keep the operational set small, track the catalog separately

The operational corpus — the slice that is actually wired into our training launchers — is eight C/C++ repositories cloned shallow and pinned to explicit refs: LLVM at `llvmorg-19.1.0`, Boost at `boost-1.86.0` with submodules, the Linux kernel at `v6.10`, fmt at `11.0.0`, googletest at `v1.15.0`, abseil-cpp at tip, folly at tip, and grpc at `v1.67.0`. Together they total about 15 GB on disk after shallow clone, all license-clean, no credentials needed. We chose exactly these eight because between them they cover the shapes of C++ that production teams ship: low-level systems C, template-heavy generic C++, modern application C++, and service-framework C++.

Separately, we maintain a much larger catalog of 142 repositories in 16 categories — OS kernels, compilers and runtimes, databases, networking stacks, browsers, game engines, the GNOME and KDE ecosystems, ML/scientific libraries, crypto, and embedded RTOSes. Each entry is tagged by on-disk size bucket (`S`/`M`/`L`/`H`) so we can budget ingestion when a future specialist needs a domain we do not yet cover. The catalog also documents the awkward sources — SQLite's Fossil repo, Chromium and V8 on a hosted source mirror, VLC and x264 on VideoLAN GitLab, Unreal requiring an Epic-linked GitHub account — so a future expansion does not re-discover the same infrastructure traps.

The split between *operational* and *catalog* is deliberate. Trying to ingest 142 repos on day one would have meant fighting infrastructure issues instead of debugging the pipeline. We did the opposite: a tiny, reproducible operational set first; everything else is metadata.

## Pipeline shape: five stages, transitional middle

The data build is orchestrated by the public data-preparation pipeline and has five stages: download, tokenize, format, cache, verify. Stage 1 shallow-clones the eight repos into a raw source staging tree under the configured data root. Stage 2 runs a libclang-based semantic indexer over each project, emits enriched text records at one document per semantic chunk (up to 4096 tokens), tokenizes with our hybrid BPE, streams into parquet shards of 50,000 docs each plus a validation parquet shard, and drops a completion sentinel. Stage 3 converts parquet into Megatron's `.bin`/`.idx` format using `uint32` token IDs (vocab is 131,072, so `uint16` is invalid). Stage 4 memory-maps the artifacts and reports doc/token counts. Stage 5 verifies that the indices parse, that no token exceeds the vocab size, and that document zero round-trips through the tokenizer.

Underneath that user-facing pipeline, we run multiple coexisting *producer* paths because we are still migrating. The current mainline producer is a semantic chunker plus libclang indexer plus enrichment jobs, which write enriched records that become parquet. The legacy flat-text producer still exists and is still used by some research lanes. There is also a binary token dataset path that historical training jobs used. Treating the consumer-side parquet contract as authoritative — and treating producer rollout as transitional — has been one of the most useful framings we adopted, because it lets us ship loader changes without waiting for the producer story to fully converge.

A subtlety worth surfacing: chunk-generation dataset names like `4k`, `8k`, `16k`, `64k`, `128k` should be read as **target buckets**, not as hard guarantees of real tokenizer-measured lengths. Several legacy chunkers still budget by a chars-per-token heuristic heuristics; only the strict producer lanes have exact-token-budgeted surfaces. We have made the mistake of assuming a `4k` shard is genuinely ≤4096 tokens under the current tokenizer, packed it that way, and then watched the loader silently crop. The fix was discipline: do not over-generalize the `chars/4` heuristic, and re-measure with the current tokenizer when in doubt.

## Filtering: language, quality, license, secrets

Every file goes through a filter stack before it reaches the chunker. Language detection drops anything whose extension or shebang is not C, C++, or a recognized header. A quality pass removes auto-generated files (large blocks of identical-shaped lines, files with absurd line lengths, embedded base64 blobs, generated lex/yacc output that adds noise without teaching the model anything). License headers are scanned to keep the corpus to permissive and weak-copyleft licenses for the libraries; the Linux kernel headers carry GPL-2.0, which we accept knowingly and tag in the per-document metadata so downstream training mixes can choose to include or exclude them.

PII and secret scrubbing runs before tokenization, not after. Email addresses become the synthetic placeholder `<redacted-email>`, network addresses become `<redacted-address>`, high-entropy strings that look like API keys become `API_KEY_REDACTED`, and any absolute user paths that survive into source comments are normalized to `<redacted-path>/`. Buckets and project identifiers in build scripts are normalized to `<bucket>` and `<project>`. We rely on community tooling here rather than rolling our own. The line we hold is "no new dependency unless we have measured it"; we measured these.

## Deduplication: file-level exact, then near-dup MinHash

Our dedup runs in two stages. Stage one is exact file-content dedup keyed by SHA-256 of the normalized text. This is cheap and catches the obvious duplicates from vendored copies of the same library inside multiple repos (Boost shows up inside other projects; abseil shows up vendored inside grpc). Stage two is near-duplicate dedup using MinHash with the BigCode dataset script. Near-dup matters more than people expect: we have seen the same algorithm implementation appear with trivial whitespace differences across dozens of repos, and feeding all of them to the model just teaches it that one specific implementation is the universe.

An honest mistake we made: in an early pipeline run we deduped *after* chunking. That produced two pathologies. First, the same function would survive in two near-identical chunks because the surrounding lines differed slightly. Second, dedup ratios looked artificially good because chunk boundaries shifted enough to fool MinHash on otherwise identical files. We moved dedup back to file-level before chunking, and the post-dedup token count dropped by another ~12% with no quality loss on our evaluation set.

## Tokenization in one paragraph, because it gets its own post

The tokenizer is a hybrid: a hand-curated fixed vocabulary for C++ primitives (special tokens, keywords, multi-character operators, preprocessor directives, single-char punctuation, common STL identifiers, small integers, diff markers, structural whitespace) merged with a learned BPE layer trained on the C++ corpus. The current shipped artifact is v3 with a 131,072-token vocabulary; the migration story from v2 to v3 — what we proposed, what we measured, what we kept — is its own writeup. The relevant fact for this post is: every document gets a leading BOS token. That single decision unlocks the entire document-masking story below, with zero changes to the storage format.

## Packing and document masking

The dataloader reads parquet via PyArrow from local paths or object-store URIs, and packs documents using BOS-aligned best-fit bin packing. This is not fixed-block padded training — we deliberately mix multiple documents into the same packed row to push utilization toward 100% instead of paying explicit pad tokens. That choice is what makes long-context training tractable for us, and it is also what forces document masking to be correct end-to-end, because a 64K row commonly contains a dozen unrelated documents.

Document boundaries are inferred on the fly from input IDs: a cumulative sum over the BOS positions gives a `doc_ids` tensor that costs O(T) per batch and requires no storage-format change. The attention backends each receive this tensor in their native form: FlexAttention composes a `document_causal_mask` with the existing softcap `score_mod`; FA3 varlen converts `doc_ids` into `cu_seqlens` and runs unpadded; the SDPA fallback materializes a 2D mask but is gated to T ≤ 8192 because the O(T²) mask is unusable above that. On TPU, both Pallas FlashAttention and JAX Splash Attention accept `segment_ids` directly. Mamba layers need an extra step: the SSM hidden state must be zeroed at boundaries, and the conv1d sliding window must be masked so the kernel does not leak across documents.

The reason we care so much about this: at 4K context the contamination signal is in the noise. At 16K and 64K it is the difference between a model that reasons about a repository and a model that hallucinates dependencies between unrelated files. Our long-context validation refuses to merge a packing change unless `val_bpb` at 16K is strictly better than the same checkpoint trained without masking.

## Quality gates: the part where we catch ourselves

Every produced shard goes through automated checks before it can be promoted to a training-data path: shard count, token count, dtype check, vocab-bound check, round-trip decoding of the first N documents, schema validation for enriched columns, and a smoke training step that consumes the shard for one batch and asserts loss is finite. The schema check is strictest on the enriched parquet path because we have a contract that says: if `token_ids` and the requested token-local arrays are present and shape-valid, trust them; if they are missing, empty, or shape-invalid, fall back deterministically rather than partially trusting bad arrays. Wrong-length `doc_ids`, broken token-structure arrays, or invalid `valid_token_count` values fail closed.

We have hit every one of those failure modes ourselves. The most embarrassing was a v4 graph dataset run that produced thousands of empty files because of a JSON schema deserialization bug in the Rust enrichment binary. The pipeline ran healthy at 100% CPU for hours and emitted statistically plausible shard counts. It was the round-trip decode quality gate that caught it: the first document of each shard was empty. We fixed the deserialization, recompiled, and added an additional gate that fails the shard if the empty-document ratio exceeds a small threshold.

Another one worth recording: an early Kubernetes-based Clang indexing job sat in `ImagePullBackOff` because the worker node service account was missing the artifact-registry read role. We were producing zero data and not noticing because the pipeline-level dashboard only showed scheduled-pod count, not running-pod count. The fix was infrastructure (grant the role, re-pull the image, fifty workers transitioned to `Running`) and the lesson was process: the dashboard now alerts on running-pod count, not scheduled.

## SFT comes after, and it is its own contract

After base training completes, supervised fine-tuning consumes a separate dataset: a 2.4 GB tool-call SFT corpus of about 1.8M examples plus the original 264 MB / 180K-example SFT dataset. SFT runs are short, have their own loader path, and intentionally do not touch the base-training producer story. Mixing the two too early was a mistake we made and had to back out — base-training shards started showing the SFT prompt format leaking in, which produced a model that looked great on assistant-style evals and worse on raw next-token prediction.

## Honest summary of where we are

The consumer-side parquet contract is canonical. The five-stage public data-preparation pipeline orchestration is reliable enough that contributors run it on workstations without help. The producer-side story is still transitional in places — we have multiple coexisting paths, target-bucket naming versus exact-token-budgeted naming is not fully unified, and the enriched parquet schema is still being extended for structure-aware training. The quality gates are the moat. Every time one of them caught a problem we had not anticipated, we wrote a new gate and kept it.

If there is one thing to take from this post, it is this: the data pipeline is not a thing you finish, it is a thing you keep falsifying. The shard schema check, the round-trip decode, the bucket-vs-actual length re-measurement, the finite-loss smoke step — those are the artifacts that let us trust a 64K training run started on a Friday afternoon.

## Stage map at a glance

| Stage | Tool | Output | Quality gate |
|---|---|---|---|
| 1 download | shallow `git clone` | raw source staging tree | repo ref pinned |
| 2 chunk + tokenize | libclang + hybrid BPE | parquet shards (50K docs) | a completion sentinel |
| 3 format | parquet -> `.bin`/`.idx` | Megatron mmap files | dtype = `uint32` |
| 4 cache | mmap + count | doc/token totals | size sanity bounds |
| 5 verify | round-trip + vocab check | pass/fail per shard | empty-doc ratio threshold |

```text
Five-stage public pipeline:
1. fetch pinned public repositories
2. build semantic chunks and enriched JSONL
3. convert to parquet and packed token shards
4. validate token and document counts
5. publish reproducible training artifacts
```

## References

- the public data-status note
- the public corpus note
- the public curriculum mapping note
- the public training-data examples note
- the public doc-masking note
- the public pipeline design note
