---
title: "Code Deduplication at Scale: MinHash, LSH, and What a 142-Repo C++ Catalog Actually Looks Like"
description: "How MegaCpp deduplicates C++ at scale: shingling choices, MinHash/LSH parameters, exact-dup SHA-256, and the tradeoffs behind near-duplicate removal."
date: "2026-04-18"
tags: ["corpus", "dedup", "minhash", "lsh", "data"]
---

If you clone enough C++ repositories, you start to notice that you have cloned `json.hpp` about forty times. And `stb_image.h`. And a surprising number of ten-line `hash_combine` utilities. Before any of that becomes training tokens, MegaCpp runs exact and near-duplicate detection over it. This post is the engineering account of what we chose, what we rejected, and what the ratios look like end to end.

## Why this matters

GitHub is heavy duplication territory for prose; for C++ it is worse. Single-header libraries get copy-pasted verbatim (`json.hpp`, `stb_image.h`, `catch.hpp`, `doctest.h`, `fmt/format.h` snapshots). Vendored dependencies sit under `third_party/`, `vendor/`, `extern/`, `deps/`. Mirrors and forks explode across GNU, Apache, and Linux projects. Without dedup, the loss curve lies: the model memorizes popular libraries, and recall on novel translation units stops improving long before training does. The practical outcome is blunt: without dedup, the model memorizes popular libraries and loses generalization.

The other reason this is load-bearing is downstream: dedup ratios feed the curriculum, the canonical-representative choice feeds provenance, and the group boundaries feed the eval harness's sense of "novel" vs. "seen." A bad dedup pass shows up as eval inflation a month later, not as an immediate error.

## 1. The operational pipeline

The canonical near-dedup pipeline used in the public sample looks like this:

```
Raw files -> Shingling (5-grams) -> MinHash (128 hashes) -> LSH (20 bands x 6 rows)
                                                              |
                                                              v
                                                      Group duplicates
                                                              |
                                                              v
                                                      Keep 1 per group
```

| Parameter | Value | Why |
| --- | --- | --- |
| Jaccard threshold | 0.7 | files sharing >=70% of shingles treated as duplicates |
| Shingle size | 5 tokens, post-normalization | smaller produces false positives on boilerplate |
| MinHash permutations | 128 | standard BigCode parameterization |
| LSH bands x rows | 20 x 6 | S-curve biased toward high-Jaccard recall |
| Exact-dup | SHA-256 of normalized file | applied before MinHash |

We did not reinvent this. The tooling is `datasketch` plus the BigCode MinHash script. Rolling a custom LSH implementation too early is a good way to spend time debugging infrastructure instead of measuring dedup quality.

## 2. Shingling choices for C++

Shingling is where most of the interesting decisions live. Natural-language MinHash pipelines usually shingle on word n-grams; applying that naively to C++ is what produces a corpus full of false duplicates, because braces, semicolons, and keyword boilerplate dominate short shingles.

The rules we settled on:

- Shingle on normalized tokens, not bytes. The public sample normalizer strips leading whitespace, compresses blank lines, and preserves comments and string literals. That normalization runs before shingling so indentation-only differences do not hide duplicates.
- Treat comments and string literals as first-class tokens. We experimented with stripping them before shingling and saw the false-duplicate rate rise sharply: two distinct implementations with identical control flow but different log strings would collapse into one group. Keeping them in the shingle space costs no recall in the high-duplicate corners (vendored headers match regardless) and buys precision everywhere else.
- N = 5 tokens. Smaller n shingles (3, 4) produce false positives across idiomatic boilerplate — every for-loop header starts to look the same. Larger n (7, 8) misses near-dupes that differ by a sprinkling of inserted `const` or renamed variables.
- Do not shingle over license headers. The license-header strip runs before shingling. Otherwise the SPDX prefix dominates the first handful of shingles in every Apache/BSL file and inflates cross-project similarity.

### What we explicitly rejected

- SimHash. Attractive for its constant-time distance, but on C++ it pulls two unrelated `for (int i = 0; i < n; ++i)`-heavy files closer than it should. Shingle-bag SimHash recovers precision but erases the constant-time win.
- AST-level fingerprints. We already use Tree-sitter for chunking; using it for dedup too doubles the parse budget for marginal recall.
- Embedding-based dedup. Cost does not justify itself at this scale; cheaper MinHash catches the duplicates we care about.

## 3. Exact-dup first, near-dup second

Order matters for two reasons: cost and group quality.

Cost: exact-dup via SHA-256 over normalized bytes is linear, cheap, and removes the long tail of copy-pasted single-header libraries before MinHash sees them. On one 100K-file slice, exact-dup alone removed **1,068,487** duplicate chunks on the Rust Tree-sitter chunker output and **32,766** on the Python brace-matcher output. That is the same input corpus; the ~30x difference is a function of chunker stability, not corpus redundancy.

Cluster quality: feeding identical bytes into MinHash creates degenerate LSH bands where one massive group swallows everything matching it. Removing exact dupes first keeps the LSH index small and the bands clean.

## 4. Picking a canonical representative

Once a group is formed, the question is which copy to keep. "Random" is a bad answer: for a vendored `json.hpp`, a random pick might land in a niche repo with an odd patch, and the specialist ends up learning the niche variant.

Our canonicalization rules, in order:

1. Prefer the copy from the upstream canonical repository if one is in our corpus. `nlohmann/json` wins over any vendored copy.
2. Otherwise, prefer the copy from the larger, more-starred repo under a permissive license.
3. Break remaining ties by the lexicographically smallest `(repo, path)` pair, for determinism.

The effect is that groups resolve toward upstream identities, and provenance records which copies were deduplicated out rather than silently discarding them. A later analysis pass can still study vendoring patterns if needed.

## 5. Interaction with the Rust chunker

Dedup ratios interact with chunker choice. The Rust Tree-sitter chunker produces stable, AST-aligned chunks (p50 about 1010 tokens, p90 about 1470 on a 1K-record sample), so exact-dup over its outputs is effective. The Python brace-matching chunker is simpler but its boundaries drift; one 135K-token file leaked through the 1024-token cap in the same test, and its exact-dup yield was an order of magnitude lower.

Second-order consequence: when the chunker is stable, shingles are stable too, and MinHash Jaccard estimates become comparable across runs. With an unstable chunker, small input changes move shingles and the same pair can group differently on consecutive passes. We did not adopt the Rust chunker because of dedup, but dedup quality is why we would not switch back.

## 6. What breaks

Three recurring failure modes.

### Generated code that looks hand-written

Protobuf-generated `.pb.cc` files, tablegen output in LLVM, Qt `moc_` sources. They pass quality filters, they are not flagged by license detection, and they form enormous high-Jaccard groups that dominate bands in the LSH index. We handle them with a separate "auto-generated markers" filter (for example `// Generated by` and `DO NOT EDIT`) rather than with MinHash. If you let MinHash deduplicate tablegen output, you are deduping the wrong signal: the real duplication is at generation-input level, not at emission level.

### Amalgamation files

SQLite distributes itself as a single amalgamated `sqlite3.c`, and several other projects distribute amalgamations as conveniences. These files are unusually large and often non-representative of how the code is normally written. MinHash will cluster them mostly against themselves, but they still distort downstream token distributions. We route amalgamations to a separate ingestion flag and exclude them from the main near-dup scan.

### Submodules

The Boost super-project pulls in roughly 180 submodules under one `boost/` namespace. Without submodule-aware provenance, Boost files appear to come from one repo and MinHash reasonably concludes they are tightly related. That is fine for dedup but hostile to specialist mixing later, so we record submodule identity alongside the parent-repo commit SHA in the provenance sidecar.

## 7. Target ratios, honestly

Summarizing what is worth quoting from the pipeline numbers:

| Stage | Reduction | Source |
| --- | --- | --- |
| Raw -> exact-dedup | ~1.4x | representative ingestion-stage reduction |
| Exact-dedup -> near-dedup | additional 1.75-2.3x | representative near-dedup pass |
| End-to-end raw -> near-deduped | ~2.5-3.3x | combined pipeline effect |
| Chunker-level exact-dedup, Rust vs. Python on 100K files | 1,068,487 vs. 32,766 removals | representative comparison on the same sample |
| Cross-repo dedup recall on a broader catalog before ingestion | n/a | not measured on a promoted dataset |

The broader source list is not treated as a promoted training corpus. Publishing a full-catalog number without running the full ingestion would be a guess.

## What we kept and what we threw away

Kept: shingles on normalized tokens (n=5), comments and string literals as first-class tokens, license headers stripped before shingling, exact-dup SHA-256 before MinHash, 128 permutations with 20 bands of 6 rows, `datasketch` plus the BigCode script, the canonicalization order (upstream -> larger permissive repo -> lexicographic tiebreak), submodule-aware provenance, and a separate auto-generated and amalgamation routing.

Threw away: a combined exact-plus-near-dup pass (splitting is faster end-to-end because the LSH index becomes an order of magnitude smaller); a repository-level sweep that would drop repos whose cross-repo cluster density exceeded a threshold (Boost's submodules pushed cross-repo density high enough to flag Boost itself for deletion — kept only as a reporting tool); and a "keep both and reweight at training time" option for near-dup groups (reweighting by duplicate count is how popular-library memorization creeps back in, and the curriculum already tells the model which kinds of code matter when).

The practical directive is straightforward: do not change shingle size, LSH banding, or the order of exact-before-near without re-running the chunker-level dedup check. Do not promote a new repository into the working corpus without running the near-dup sweep against the current set. And do not collapse exact-dedup into the MinHash pass for superficial simplicity; that cost comes back as a larger index and weaker groups.

## References

- [MegaCpp public repository](https://github.com/DatasunriseOU/cppmega)
- [MegaCpp article samples](https://github.com/DatasunriseOU/site_samples/tree/main/articles)
- [The Stack](https://arxiv.org/abs/2211.15533)
- [BigCode evaluation harness](https://github.com/bigcode-project/bigcode-evaluation-harness)
- [datasketch](https://github.com/ekzhu/datasketch)
