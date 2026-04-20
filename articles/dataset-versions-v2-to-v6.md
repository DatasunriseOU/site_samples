---
title: "v2 to v6: Four Generations of the C++ Dataset, and Why We Kept Them All"
description: "What changed between v2, v3, v4, v5, and v6 of the C++ training corpus, why each step happened, why we kept the older formats backwards-compatible, and the val-bpb each one bought us."
date: "2026-04-18"
tags: ["data", "dataset", "C++", "versioning", "curriculum"]
---

The data side of MegaCpp has accumulated five labeled generations: `v2`, `v3`, `v4`, `v5`, and `v6`. None replaced the previous. They live next to each other in the object-store layout, share the same raw corpus, and are loaded by the same training loader because every step was deliberately additive at the schema level. This post is the engineering story of that progression — what each version is, what problem it solved, and why we are paying the storage and operational cost of keeping all of them. They are not model versions; they are *producer* versions over the same C++ corpus, and the decision to make them coexist instead of migrate-and-delete is the single thing that has saved us the most engineering pain across the curriculum work.

## The shared substrate

Before any of the versions, there is the corpus. Eight pinned C/C++ repositories cloned shallow — LLVM at `llvmorg-19.1.0`, Boost at `boost-1.86.0` with submodules, Linux at `v6.10`, fmt at `11.0.0`, googletest at `v1.15.0`, abseil-cpp at tip, folly at tip, grpc at `v1.67.0` — plus a 142-repo catalog tracked separately. Roughly 15 GB on disk after shallow clone. One 131,072-token hybrid C++ tokenizer. The differences between versions are how we *chunk*, *order*, *enrich*, and *resolve cross-file references* over that same set of bytes.

## v2: full files, pre and post commit

`v2` is the first dataset that was more than a flat dump. It walks the commit history of each repo and, for each commit touching a C++ file, emits the entire file before the commit and the entire file after as two documents. That is it. Raw flat files have no temporal signal — the model never sees a piece of code as something that used to look different — and file-level pairs are the cheapest possible way to get *change* into the corpus without inventing a structured format.

Raw `v2` archives total roughly a couple of TB of compressed JSONL across 27.6 M documents. It tokenizes into `uint16` binaries because, at the time, the tokenizer fit in 16 bits. That later changed; see the `v6` section.

Every later version inherits from the same commit walk; only the *representation* of each commit changes.

## v3: structured inline diffs

`v3` is `v2` rewritten as a single document per commit, where removed lines appear as C++ comments (`// Removed: ...`) and added lines appear as live code, under a synthesized file header. Two header styles coexist: `v3_doxygen` uses Javadoc-style `/** @file ... @brief ... */` headers, `v3_simple` uses plain `// File: ...` comments.

Why both styles? Because we did not know which one the model would learn faster from. Doxygen is more verbose and matches the conventions of the larger projects (LLVM, folly); simple is less noisy and avoids teaching the model to associate diff content with documentation tooling. Rather than guess, we built both, packed them at equal weight in early curriculum phases, and let eval numbers speak. They do not differ enough to justify keeping only one — and once both exist, the marginal cost is storage.

The win over `v2` is that the model sees the diff as code with a comment, in-place, in roughly the same shape as a reviewer would see it on a screen. `v2` taught "what does this file look like before and after"; `v3` teaches "what was changed, and what the change replaced." `v3` also tokenizes to `uint16`. Same archive shape, same raw commit walk underneath.

## v4: tree-sitter context graph

`v4` adds the first *graph*. For every modified function in a commit, we build a strict 64K-token window containing the target plus its direct callers and direct callees, extracted with a tree-sitter AST walker. The model sees not just the modified function, but who calls it and who it calls, in the same window.

`v4` had an embarrassing first month: an empty-file bug from a JSON schema deserialization mismatch in the Rust binary was producing zero-byte outputs for a non-trivial fraction of repos. Once fixed and recompiled, the pipeline saturated 40 cores; hundreds of repos finished cleanly. `v4` is *approximate* by design — tree-sitter does not resolve names across files, does not see overloads, does not know which `foo()` a particular call resolves to under namespaces or templates. For 16K/64K curriculum windows that is fine, and the point of `v4` is throughput: commodity CPU, no build system, no compile database, no per-project setup.

The platform-detection upgrade that landed with `v4` enrichment uses an Aho-Corasick matcher over 190+ patterns. It auto-detects 30+ OS, 12 RTOS, 16 GPU/accelerator targets, 25+ architectures, 14 compilers, and a C++-standard hint. Those land in the `platform_info` field downstream lanes either preserve or refine.

## v5: clang semantic graph

`v5` is the answer to "what does `v4` lie about?" Where tree-sitter approximates, libclang resolves. `v5` drives Clang with each project's `compile_commands.json`, walks git history incrementally so the build context is correct at every commit (we hit the bug where `cmake` was only run once at HEAD and silently used wrong flags for older commits — fixed by regenerating the CDB whenever build files change between commits), and emits 100 %-accurate semantic relationships: cross-file calls resolved through the actual frontend, types resolved through the actual symbol table.

`v5` runs on a dedicated worker fleet — historically 50 worker pods, later scaled to 80, then to 246 across the 4K/8K/16K/32K/64K/128K size buckets. Deployment took longer to stabilize than the indexer itself: image pull failures until the node service account got artifact-registry read; thread-based workers deadlocking on hung clang translation units until we switched to process-based workers with per-commit lifecycle; giant working trees stalling under checkout-heavy history walks until we rewrote the per-commit extraction around direct object reads.

`v5` produced the data that lives in production today as `clang_semantic_4k_v10`. The `v10` suffix is a producer revision counter, not a schema generation — schema-wise it is still `v5`. It mixes at 0.6 against `clang_commits_4k_v1` at 0.4 in the production launchers; that is the data the long-context specialists learn cross-file repository reasoning from. Cost: tens of GB of remote output across the six size buckets, weeks of operational tuning, and a non-zero fraction of repos we had to give up on (embedded RTOSes with build systems we cannot reproduce stay on `v4`). Benefit: the only producer in the stack whose call edges we actually trust.

## v6: enriched parquet

`v6` is the version where the dataset stops being just text. Same commit walk underneath. Same `v5`-quality semantic edges where available, `v4` tree-sitter edges as fallback. The change is the *schema*: each parquet record now carries dense structural metadata as additional columns.

The full enriched contract is larger than this summary, but the columns that matter are:

- `text` — unchanged. Backwards-compatible.
- `structure_ids` — `list<u8>` of length `len(text)`. Per-character category, one of nine: `other`, `preamble`, `func_sig`, `func_body`, `class_decl`, `class_member`, `comment`, `typedef`, `namespace`.
- `chunk_boundaries` — `list<struct {start, end, kind, name, dep_level}>`.
- `call_edges` — `list<struct {from, to}>`. Intra-document call edges. Indices into `chunk_boundaries`.
- `type_edges` — `list<struct {from, to}>`. Type dependency edges.
- `ast_depth`, `sibling_index`, `ast_node_type` — optional per-character AST metadata from tree-sitter.
- `platform_info`, `language_info`, `build_info` — optional per-document metadata.

`text` is byte-identical to what a `v2`-era training loader would have read. That is the whole reason the format ships. A naive dataloader sees text and ignores the rest; the structure-aware loader feeds `structure_ids` and `dep_level` as input embeddings and (in the original Variant C design) uses the edge tables for relation bias. Same shard, two consumers.

One `v6` schema decision paid for itself many times over: top-level metadata may be summary-grade, with detailed per-source provenance in a `constituent_provenance` sidecar. Producers do not all emit identical Arrow layouts — some enriched paths write char-level structure plus `platform_info`/`language_info`; clang-resolved paths may also emit `build_info`; some lanes use Arrow structs, others JSON sidecars. The authoritative behavior is the consumer-side tolerant loader, not any single producer's schema.

There was also a tokenizer-driven jump under `v6`: the production tokenizer is now 131,072 entries, exceeding `uint16`. Pretokenized `v6` shards switched to `uint32` token IDs (covered in the binidx-pipeline post). Older `v2`/`v3` `uint16` archives remain valid against the older tokenizer; the new ones are not.

## Why everything is backwards compatible, on purpose

Migrating a corpus across schema boundaries is the single most expensive thing you can do in a data stack. Re-tokenizing 27.6 M documents because you renamed a column burns a week of compute time and a week of human attention. Every version after `v2` was designed so the previous version's consumers do not break. `v3` is `v2` re-rendered against the same commit data. `v4` adds graph outputs in a separate prefix. `v5` writes clang-resolved semantic outputs parallel to the tree-sitter outputs, and mixed training pulls from both in a weighted launcher mix. `v6` is the one that actually had to share a schema with its predecessor — and it solved it by making `text` the only required column and everything else optional, absorbed by the dataloader's tolerant normalization. The cost of this discipline is storage and a longer producer matrix; the benefit is that we have not had to do a single hard re-index in two years.

## Val-bpb story, honestly

The honest version of "what each step bought us in val-bpb" has two halves: numbers we measured cleanly, and numbers we did not.

Phase 1 (4K context, syntax mastery from `v2`/`v3` packed into 4096-token sequences over `uint16` binaries) is the regime where val-bpb falls fastest. Production numbers land between roughly **0.466** and **1.2** depending on variant, but those are *model-config* deltas, not dataset-version deltas — we never ran a clean ablation swapping only dataset version at scale. Phase 2 (16K, file-level) is where the tree-sitter context graph started to matter; Phase 3 (64K, repository reasoning) is where the clang-resolved graph showed up most clearly, and the long-context specialists do not exist as a useful product without it. Phase 4 (structure-aware over `v6` enriched parquet) overlaps Phases 2 and 3 because `v6` is backwards compatible; the relevant ablation is "same compilable C++, with vs without `structure_ids`/`chunk_boundaries`/edges." Where we have measured directly, it improves loss; we will not put a single number on it because the delta varies by sparse/hybrid configuration and we do not want to fish for the most flattering one.

The thing we do *not* claim: that a clean ablation between `v2` and `v6` at fixed model config exists in our production logs. It does not. The corpus co-evolved with the model, curriculum, and tokenizer. What we have is operational evidence that each generation unblocked something the previous could not: `v2` unblocked temporal signal, `v3` unblocked diff-shape learning, `v4` unblocked cheap graph context, `v5` unblocked correct cross-file edges for long-context lanes, and `v6` unblocked structure-aware training without a forced re-tokenization.

## What we keep, what we throw away

`v2` and `v3` archives stay - the cheapest reproducibility insurance, and they keep `v2`/`v3`-era checkpoints replayable. `v4` outputs with a corresponding `v5` replacement get garbage-collected aggressively; the rule is to never delete a producer output unless the remote copy verifies complete, the file count matches, and no consumer still reads the local copy. `v5` outputs are sacred until a revisioned successor under the same schema publishes. `v6` is what every new ablation runs against by default.

## What we got wrong

The version numbers are confusing: `v2` through `v6` are schema generations, but each also carries producer-revision counters that are not the same thing. If we did this again, schema generation and producer revision would be on different axes from day one.

Letting producers coexist for months turned "which producer emits what" into a cross-reference exercise. The fix is to sunset the older producer deliberately once migration completes, not to leave the transitional middle open indefinitely.

We initially thought we could collapse `v4` and `v5` into a single graph layer. We cannot: `v4` is a throughput layer, `v5` is a correctness layer. Every collapse attempt reinvented the same split under different names.

## What each version is for

| Version | Granularity | Producer surface | Curriculum slot |
|---|---|---|---|
| v2 | file pre/post-commit | flat JSONL | Phase 1 (4K) |
| v3 | structured diff | simple / doxygen header styles | Phase 1 (4K) |
| v4 | tree-sitter graph window | AST-aware chunker | Phase 2 (16K) |
| v5 | clang-resolved graph | libclang indexer | Phase 3 (64K) |
| v6 | enriched parquet + structure | enriched producer | Phase 4 (structure-aware) |

```text
Coexisting object-store layout:
- v2: JSONL archives
- v3: JSONL archives with multiple header styles
- v4: tree-sitter graph parquet shards across multiple context buckets
- v5: clang-resolved graph parquet shards
- v6: enriched parquet shards
```

## References

- [MegaCpp MegaCpp sample pack articles directory](https://megacpp.com/blog)
- [MegaCpp MegaCpp sample pack docs directory](https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md)
