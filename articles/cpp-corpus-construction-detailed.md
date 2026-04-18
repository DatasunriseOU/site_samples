---
title: "How We Actually Build the C/C++ Corpus: Eight Pinned Repos, 142-Repo Catalog, and the Filters Between"
description: "Detailed construction of the MegaCpp C/C++ training corpus: eight pinned operational repositories that reach training, the 142-repo extended catalog we draw from, the filters and normalization passes in between, and the awkward sources that force special-case handling."
date: "2026-04-18"
tags: ["corpus", "data", "c++", "filtering", "provenance"]
---

Every time someone asks "what is MegaCpp trained on," the honest answer has two halves. Half one: eight public C/C++ repositories, pinned by tag, shallow-cloned, totalling roughly 15 GB on disk. Half two: a 142-repo catalog across sixteen categories that we treat as pre-vetted inventory for future specialists, plus a filter stack, a normalization pass, and a handful of ugly sources that do not live on GitHub at all. This post walks through both halves and the seams between them.

## Why MegaCpp cares about this

A C++ specialist family stands or falls on what bytes it saw. We can pick any optimizer, any attention backend, any parallelism split, and it will not matter if the corpus leaks Python from vendored scripts, confuses tab-indented kernel code with two-space modern C++, or silently absorbs auto-generated lex/yacc output at scale. The corpus is the product; the training code is plumbing. We kept the corpus small, pinned, and auditable, and kept a separate catalog for everything we consciously have not ingested yet. "We trained on permissively licensed C++" is defensible only if you can name, for each file, which repository it came from and at which commit — and that audit has to run on a workstation.

## What we built in the POC

The operational list lives in the public data-preparation notes and the download stage of the corpus pipeline. Eight repos, each pinned:

1. LLVM at `llvmorg-19.1.0` — modern C++, compiler workings, MLIR, LLDB, test suites.
2. Boost at `boost-1.86.0` with submodules — template-heavy generic C++ across about 180 libraries.
3. Linux at `v6.10` — C, systems, the kernel header surface.
4. fmt at `11.0.0` — small, high-quality modern C++.
5. googletest at `v1.15.0` — test patterns and macros.
6. abseil-cpp at tip — general-purpose C++ commons.
7. folly at tip — large-scale C++ utilities.
8. grpc at `v1.67.0` — a large real service framework.

Combined, this is about 15 GB on disk after shallow clone, all license-clean under the Apache/BSL/MIT set above, with the Linux kernel treated as GPL-2.0 and tagged per-document. Shallow clones pin the exact commit via the remote tag; full reproducibility requires mirroring the raw corpus tarball, which we do for release candidates but not every dev laptop.

The extended catalog lives in the public corpus catalog: 142 entries across sixteen categories — OS kernels, compilers and runtimes, build systems, databases, networking and web servers, browsers, graphics and game engines, multimedia, desktop toolkits, GNOME, KDE, editors and office suites, ML and scientific libraries, crypto, C++ library ecosystems, and embedded/RTOS. Each row carries a size bucket (`S`/`M`/`L`/`H`), a canonical URL, and a note when the source is awkward. Nothing in the catalog is wired into training. It is a menu.

### Stage 1: shallow download

the public download stage clones the eight operational repos into the raw source staging tree under the configured data root. It is idempotent — re-running skips existing directories. Tags are pinned per repo; `--recursive` on Boost is mandatory because of the submodule-per-library layout, and a flat clone silently produces an almost-empty tree.

### Stage 2: filter, index, tokenize

A file enters the pipeline only if it survives the filter stack:

1. Extension and shebang check. Recognized C and C++ translation units and headers survive. Scripting files, Markdown, reStructuredText, build metadata, and vendored third-party configs are dropped before the indexer ever sees them.
2. Auto-generated content detection. Long runs of identical-shaped lines, absurd line lengths, embedded base64 blobs, and generated lex/yacc output fail the quality gate. Including these makes the model better at imitating them and worse at writing C++ a human would write.
3. License header read. Every file carries its repository's license forward through the pipeline; per-document metadata records the SPDX identifier when present. GPL-2.0 content from the Linux kernel survives but is tagged for downstream mixes.
4. PII and secret scrubbing. Email addresses normalize to the synthetic placeholder `<redacted-email>`, IPs to `<redacted-network-address>`, high-entropy blobs that look like keys become `API_KEY_REDACTED`, and stray user paths collapse to `<redacted-path>/`. This runs before tokenization, not after, because after is too late.

The surviving files are handed to a libclang-based semantic indexer, which emits enriched text records at one document per semantic chunk, bounded at 4,096 tokens per chunk. Each document carries the indexed symbol surface for the chunk, so downstream enrichment (call edges, type edges, structure IDs) has something to compute against.

The tokenizer itself is the hybrid fixed-vocab-plus-BPE artifact (the public tokenizer sample), covered in its own deep dive. For corpus construction it suffices to note: it handles C++ morphology, treats multi-space and tab indent as first-class tokens, keeps comments intact instead of stripping them, and guarantees a BOS token at the head of every document. That last property is what enables the document-masking story downstream.

The tokenizer writes parquet shards of 50,000 documents each plus a the validation parquet shard, and drops a a completion sentinel when the repo is done. Crash-resumable by construction: a crashed repo can be re-run, a completed one is skipped.

### Stages 3-5: format, cache, verify

Stage 3 converts parquet to Megatron's `.bin`/`.idx` layout with `uint32` tokens (vocab 131,072, so `uint16` is invalid). Stage 4 memmap-validates. Stage 5 asserts the indices parse, that no token exceeds the vocab bound, and that document zero round-trips through the tokenizer. Any failure is fatal; we do not ship silent fallbacks at the verify stage.

### Semicolons, comments, encoding: the unsexy normalization

Three decisions deserve to be called out because they changed downstream behavior materially.

Semicolons are never stripped and never normalized. They are load-bearing in C++ and the tokenizer treats them as punctuation. Early ablations that aggressively collapsed whitespace around semicolons produced a model that could not re-emit them in the right places; we reverted fast.

Comments are not noise. Comment bytes are about 24% of the reference corpus by size, and 15% of all lines. The tokenizer explicitly invests in common comment patterns (`//`, `/* */`, Doxygen `* ` continuations). Stripping comments is a trap: it disconnects the code from the explanations engineers wrote next to it, and those explanations are often the training signal we want most.

Encoding is almost boring. The measured corpus is over 99.98% ASCII. Non-ASCII bytes are 0.020% of the total, non-English comment content appears in roughly 3.4% of files (mostly accented author names and the occasional Unicode math symbol), and UTF-8 multi-byte characters are negligible. We considered a Unicode-to-English normalization pass and dropped it: the benefit is below the cost of introducing a non-invertible transform in a pipeline that otherwise round-trips cleanly. Standard UTF-8 byte fallback in the BPE covers the remainder.

The shape of indentation also mattered more than we expected. Four-space indent is 31% of indented lines, two-space is another 31%, tabs are 26%, eight-space is 12%. None of these can be ignored. The tokenizer carries explicit tokens for `"  "` (two-space), `"    "` (four-space), `"        "` (eight-space), and `"\t"`, and the filter stack does not normalize one style to another. Training the model to be flexible across indent styles is a stated goal, not a bug.

### Awkward sources: Fossil, Chromium, VideoLAN, Unreal

Some of the catalog's most valuable entries do not live on GitHub at all. SQLite is Fossil-hosted and ingested via its release amalgamation. Chromium, V8, and Fuchsia are googlesource monorepos fetched via `depot_tools`. VideoLAN (VLC, x264) runs its own GitLab; x265 is on Bitbucket; Mesa 3D is freedesktop GitLab; Eigen, Inkscape, GnuTLS are on gitlab.com; BoringSSL is googlesource; WireGuard is `zx2c4.com`. Unreal Engine is GitHub but gated on an Epic-linked account, so it stays in the catalog with an access note rather than the operational list. GNOME and KDE GitHub orgs are read-only mirrors; provenance must point at the upstream GitLab.

Two additional footnotes for the runbook: Boost must be cloned `--recursive` or you pick up almost nothing; and the Linux kernel is scoped to headers and specific subsystems rather than the full source, both for license reasons and because full-kernel-C dominates every corpus statistic it touches.

## How it lands in MegaCpp

In the production stack, the five-stage dataset preparation workflow is the interface. It runs on a dev workstation end-to-end, which is the property we care about most: a contributor with a clean clone and the right environment variables can rebuild the reference corpus from scratch without asking anyone for anything. That is non-negotiable for reproducibility claims.

The tokenizer artifact is not vendored inside the production repo. The tokenizer build is owned upstream, and stage 2 of the production pipeline points at a pinned local build via configuration. Duplicating the tokenizer into the production tree would mean maintaining two copies of thousands of lines of active tokenizer code; we chose the dependency instead.

Training launchers consume the Megatron `.bin`/`.idx` pair by path, with an explicit split (typically `98,1,1`), the vocab size, and the tokenizer directory. The launcher does not care how the corpus was built; it only cares that the files pass stage 5's invariants. That separation is what lets us iterate on the producer side without invalidating every training launcher.

## Ablations and what we kept

The ablations that shaped the current pipeline are engineering, not research:

- We tried aggressive deduplication on the reference corpus early on, hashing by file body. It removed genuine near-duplicates — header/source template pairs — that the model actually benefits from seeing together; we reverted to an "exact duplicate only" policy and let the chunker handle the rest by semantic boundaries.
- We tried stripping SPDX headers to save context. It is true that inline license headers waste tokens. It is also true that a specialist trained without any SPDX exposure later failed to emit SPDX headers on its own output; we kept the headers in the text and tagged the license separately in metadata.
- We tried a unified chars-to-tokens heuristic (`chars/4`) in place of running the tokenizer on every chunk boundary. Fast, and wrong, and the loader silently cropped. The producer paths that still use it are now clearly marked as "target bucket, not guaranteed length," and the consumer-side tokenizer re-measures on anything labelled `4k`, `8k`, `16k`, `64k`, or `128k`.
- We tried bringing the full Linux kernel in as a single mass. It dominated corpus statistics and dragged the model toward kernel-C idioms at the expense of modern C++. Scoping it to headers and specific subsystems preserved the exposure without the distortion.
- We tried ingesting the full catalog into training directly. We did not finish that ingestion because the return per repo fell off fast after the first eight; the working set already covers the four shapes of C++ that matter (low-level systems C, template-heavy generic C++, modern application C++, service-framework C++) and additional catalog repos add duplication more than diversity.

## Production checklist

- Every operational repo is pinned by tag or specific commit, never tracked as a floating branch.
- The raw source staging tree is scoped to license-clean public C/C++ and is versioned as a raw blob for release candidates, because upstream tags can be re-cut.
- Stage 5 (the dataset verification stage) must pass before a build is promoted. Silent fallbacks are disallowed.
- Per-document metadata records the source repository, the upstream license identifier, and a normalized path fragment that does not leak local filesystem roots.
- SHA-256 of the produced `.bin` is recorded in the experiment log alongside the training checkpoint.
- The tokenizer artifact is tied to a specific upstream commit; that commit hash is recorded with every checkpoint the dataset feeds.
- Non-GitHub sources (SQLite Fossil, Chromium and V8 and Fuchsia on googlesource, VideoLAN and Mesa GitLab, zx2c4.com, Epic-gated Unreal) are scripted individually and never assumed to be reachable via a default clone URL.
- PII and secret scrubbing runs before tokenization, not after. An audit finding post-training is a far worse outcome than a tokenizer miss.
- Comments are preserved. Encoding is preserved. Indentation style is preserved. Anyone proposing to normalize any of these brings ablation numbers first.

## Corpus slice snapshot

| Slice | Source family | Filter | Share |
|-------|---------------|--------|-------|
| Core C++ | permissive OSS | license allow-list, dedup | majority |
| Headers and stdlib | curated | include-graph gated | moderate |
| Tests and fixtures | OSS test dirs | doc-mask, PII scrub | small |
| Commentary and RFCs | documentation trees | markdown-only | small |

```yaml
# per-slice policy the corpus builder consumes
slices:
  core_cpp:
    license_allow: [MIT, Apache-2.0, BSD-3-Clause, ISC]
    dedup: minhash-lsh
  headers:
    gate: include_graph_reachable
  tests:
    scrub: [email, ipv4, high_entropy]
```

## References

- the public data-preparation notes
- the public corpus catalog
- the public curriculum mapping note
- the dataset preparation workflow
- the download stage of the corpus pipeline
- the tokenization stage of the corpus pipeline
- the formatting stage of the corpus pipeline
- the dataset verification stage
- the public tokenizer sample
- the public vocabulary analysis note
- [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — Tao et al., NeurIPS 2024]
