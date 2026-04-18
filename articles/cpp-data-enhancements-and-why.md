---
title: "Data Enhancements: Why structure_ids, AST Features, and the Clang Graph Are Worth Their Cost"
description: "The structural metadata layered on top of raw C++ source - structure_ids, chunk_boundaries, call_edges, type_edges, tree-sitter AST features, and the optional libclang semantic graph. What each one is for, the ablation that justified shipping it, and what we pay in storage and runtime."
date: "2026-04-18"
tags: ["data", "enrichment", "tree-sitter", "clang", "ablation"]
---

Raw C/C++ source is sequence-of-bytes, which is what every general-purpose language model has to make do with. We pay extra to do better. The enriched parquet schema annotates each token with five pieces of structural metadata, every chunk with a typed boundary plus a dependency level, and every document with two graphs (call edges and type edges). On top of that, the long-context lanes consume an optional libclang semantic graph that resolves what tree-sitter can only approximate. This post is the engineering case for each of those layers: what it is, why we added it, what the ablation said, and what it costs.

## Why MegaCpp cares about this

Code is a structured artifact pretending to be flat text. A function body is a span. A class declaration carries an implicit dependency on its base class. A `#include` is a routing hint. A call from `f` to `g` is the strongest signal in the file about what `f` does. None of that is in the bytes - it is recoverable, but only if you spend compute outside the model to parse it. The choice we made is to spend that compute once, offline, materialize the result into the parquet record as token-aligned columns, and let the model read it for free at training time.

The alternative is to make the model rediscover this structure from raw bytes. Some of it does, eventually. The empirical observation in our ablation logs is that the structure-aware path consistently wins on val_bpb at fixed compute, and the throughput cost - once we engineer it correctly - is smaller than the win. The interesting engineering is in the qualifier "once we engineer it correctly." The naive implementation is several times slower than the model and a graph cache nightmare on TPU.

## What we built in the POC

Five distinct enhancement layers, each with its own producer, consumer, and ablation history.

**Layer 1: per-token `structure_ids`.** Nine `StructureCategory` entries (`OTHER`, `PREAMBLE`, `FUNC_SIGNATURE`, `FUNC_BODY`, `CLASS_DECL`, `CLASS_MEMBER`, `COMMENT`, `TYPEDEF`, `NAMESPACE`) produced by the semantic chunker as per-character category bytes and materialized to token level via first-character sampling. The consumer is `StructureEmbedding`: a 9-entry learned table projected through a low-rank bottleneck (default `bottleneck_dim=64`), zero-initialized so the path is identity at start.

**Layer 2: chunk boundaries with dependency levels.** A `chunk_boundaries` column of `list<struct {start, end, kind, name, dep_level}>`. The chunk kinds are the same nine categories as `structure_ids`, lifted to the chunk level so a whole function body is one chunk rather than per-token bytes. `dep_level` is the BFS depth of the chunk in the document's local dependency graph (a function called only by main is at dep_level 0; a helper called by that function is at dep_level 1; and so on). The consumer feeds `dep_level` as a second structure embedding, and the chunk boundaries themselves drive the TreeFFN message-passing path.

**Layer 3: call_edges and type_edges.** Two `list<struct {from, to}>` columns, indices into the chunk array. These are what make the document a graph instead of a sequence. They originally fed a learned per-head relation bias (`RelationBiasComputer`) that added a chunk-pair bias term to the attention logits. We removed that class in March 2026 (see ablation notes below) but kept the edges in the parquet because they still drive the TreeFFN path and the offline analytics; their cost is small and the optionality is cheap.

**Layer 4: tree-sitter AST features.** Three per-token columns : `ast_depth` (clamped to 63), `sibling_index` (clamped to 63), and `ast_node_type` (256-bucket categorization via a shared node-type map). Buckets group semantically similar nodes (declarations 1-9, statements 10-19, expressions 20-29, types 30-39, literals 40-49, operators 50-59, and so on). The producer and consumer agree on numbering so producer paths stay in sync. The consumer is the same `StructureEmbedding`, with three extra sub-tables folded into the stacked lookup.

**Layer 5: the libclang semantic graph (optional).** This is the only layer that requires a real C/C++ frontend instead of an AST parser. The semantic indexer drives libclang against each project's `compile_commands.json`, walks the AST through the proper translation unit, and emits cross-file resolved relationships: which symbol does this `foo()` call resolve to under namespaces, overloads, and templates; which type does this declaration depend on through the actual symbol table. The output replaces tree-sitter's intra-document `call_edges`/`type_edges` with cross-document, semantically resolved edges plus per-file `compile_commands` provenance under `build_info`. This is what feeds the long-context curriculum lanes; tree-sitter graphs are fine at 16K, the libclang graphs are what justify the 64K and 128K budgets.

On the consumer side, `PlatformEmbedding` is a related-but-separate enhancement: a per-document multi-hot platform-label embedding (`nn.EmbeddingBag` over a 113-entry vocabulary covering OS, GPU, and architecture targets) that broadcasts a single vector across the sequence. The platform labels are produced by the platform scanner described in the public data-preparation notes (~1640 lines, ~190 patterns covering 30+ OS, 12 RTOS, 16 GPU/accelerator targets, 25+ architectures, 14 compilers, plus a C++-standard hint). It is not strictly part of the structural enrichment, but it ships through the same parquet contract and falls back zero-initialized like the rest.

## How it lands in MegaCpp

The producer side ships intact. MegaCpp calls into an upstream semantic chunker and semantic indexer at a pinned revision; vendoring those binaries would mean owning several thousand lines of actively maintained Rust and libclang glue code, and the dependency is the smaller cost. The schema is canonical, and the consumer-side tolerant loader is the source of truth for what a record can look like.

The consumer side is where MegaCpp makes choices. `StructureEmbedding` lifts as-is, including the stacked-embedding bottleneck optimization and the `active_components` selector (so a deployment can ship `core` - structure plus `dep_level` only - without paying for AST features). `PlatformEmbedding` lifts as-is. `RelationBiasComputer` was dropped in March 2026 across all presets: at scale it did not improve over `structure_emb` + `tree_ffn`, and it cost test surface, auto-fit accounting, and FSDP wrapper code. Edges remain in the parquet because they feed `tree_ffn`.

The TreeFFN message-passing module is being moved behind a feature flag that defaults on for the structure-aware presets and off for the dense reference. It is what consumes `chunk_boundaries`, `dep_levels`, and the edge columns to compute a per-chunk message-passed representation that gets pooled back to per-token. We are not lifting it as-is to a kernel path - we tried, and the per-batch chunk topology is too irregular to compile usefully. The TPU fast path in `StructureGraphEnricher.forward` is a bmm-based pooling plus cumsum-based top-K neighbor selection that avoids the `scatter`/`searchsorted`/`topk`/`nonzero` ops that lower badly on XLA. That fast path is what makes the structure features near-zero-cost on TPU at our hybrid-preset scale.

## Ablations and what we kept

The R6 ablation matrix from March 2026 is the canonical receipt. Same model geometry, same step count, same hardware tier, varying only the feature stack. Throughput (median tok/sec) and Val BPB at step 50:

- R6-E (bare baseline, no structure, no enriched): 595,585 tok/sec, 2.493 BPB
- R6-A (DSA + Engram + mHC, no enriched data): 496,962 tok/sec, 2.299 BPB, -17% throughput
- R6-F (full: enriched + structure_emb + tree_ffn + DSA + Engram + mHC): 316,047 tok/sec, **2.287 BPB**, -47% throughput

R6-F is the best quality at the worst throughput. R6-A is the practical sweet spot. R6-F minus R6-A (the marginal effect of enriched data plus structure features) is a small BPB improvement at 36% throughput cost. We kept the structure-aware path because the BPB win compounds across the curriculum and the throughput cost is recoverable through engineering. The pretokenized-vs-char-level fix recovered roughly 4x in the dataloader (4,172 to 17,297 tok/sec). The eager-segment-materialization fix recovered another 2.4x (562 to 1,324 tok/sec). The conv1d-vs-Python-loop fix recovered 22% on the Mamba doc-mask path. The structure_emb bottleneck dim 64 recovered 23% on the structure-emb-enabled runs.

The accelerator-kernel breakdown told a more flattering story than R6 alone. The full-feature `profile_enriched` run (depth-52 hybrid preset, 4.1B, all features, 75.0s self CUDA) versus a `profile_minimal` run (972M, 8 experts, no enriched) came out at -0.6s difference between enriched and the closest mod-bypass variant - within noise. The conclusion in the changelog is explicit: enriched features (embeddings, tree_ffn) are invisible in the profile, lost in noise, not a bottleneck once the loader-side bottlenecks are fixed.

The features that did not survive: `RelationBiasComputer` was removed because at scale it did not improve over the `structure_emb` + `tree_ffn` combination. The relation-bias experiments stayed promising in synthetic benchmarks but consistently failed to add meaningful BPB on real corpora once the other paths were enabled. Lesson recorded as a directive in the cleanup commit: do not re-add a learned per-head pair bias on top of TreeFFN without an ablation that shows it beats the cost.

The features that we ship but that are still under evaluation: the `ast_depth` / `sibling_index` / `ast_node_type` columns. They land in the parquet, they feed `StructureEmbedding` when `active_components="all"`, and we have receipts that the configuration trains stably. We do not have a clean ablation showing they add BPB independently of `structure_ids` and `dep_levels`. The cost is small - three extra `uint8` arrays per document and three more sub-tables in the bottlenecked embedding lookup - and removing them later is a one-line config change. They stay opt-out, not opt-in.

The optional libclang semantic graph is the most expensive enhancement: it needs `compile_commands.json` per project, libclang, a real build environment, and incremental history walking. We fixed the obvious bugs: the compilation database being generated only once at HEAD and then reused with stale flags for older commits, thread-based workers deadlocking on hung clang translation units, and giant working trees stalling under full checkout-based history walks. The fixes were per-commit regeneration, process-based workers, and direct object extraction instead of heavyweight checkout churn. The operational overhead is real; we keep it because it is the only producer whose call edges we trust at long context.

## Production checklist

- Every enhancement column is optional in the loader. Missing or shape-invalid columns fall back to deterministic defaults; partial trust is a correctness bug.
- `StructureEmbedding` and `PlatformEmbedding` are zero-initialized. Adding them to an existing checkpoint must produce zero contribution at start.
- The stacked-embedding bottleneck (`bottleneck_dim=64`) is on by default for structure features. Without it the path is 23% slower for no quality gain.
- The pretokenized columns are mandatory in the production data path. The char-level columns exist only as input to the offline materializer.
- `lazy_enriched_segment_materialization=False` whenever both `tree_ffn` and `relation_bias` are enabled - this gating recovers a 2.4x throughput regression and is non-negotiable for the full-enriched config.
- The TreeFFN TPU fast path is bmm-based pooling plus cumsum-based top-K. No `scatter`, no `searchsorted`, no `topk`, no `nonzero` in the structure consumer. Each one breaks the XLA static-shape contract.
- The libclang lane uses `ProcessPoolExecutor`, not `ThreadPoolExecutor`, with per-commit lifecycle.
- The libclang lane regenerates `compile_commands.json` whenever build files change between commits. Once-at-HEAD is silently wrong for older commits.
- The Aho-Corasick platform scanner is fast enough to run inline; do not move it offline as an "optimization."
- The chunk-kind whitelist for chunk-level dedup is a different list from the chunk-kind set the model consumes through `chunk_boundaries`. Do not collapse them.
- The `RelationBiasComputer` path is removed and stays removed unless an ablation shows it beats `structure_emb` + `tree_ffn` at real scale.

## Enhancement snapshot

| Layer | What it adds | Why it matters |
|-------|--------------|----------------|
| Doc-mask | loss mask over prose/comments | prevents natural-language overfit on code loss |
| Include graph | cross-file context | header-aware completion quality |
| Semantic blocks | function/class boundaries | long-context packing without chopping symbols |
| Compile probes | sanity builds on a fraction | catches corpus-wide syntax regressions |

```yaml
enhancements:
  doc_mask: { weight: 0.1, scope: ["comments", "markdown"] }
  include_graph: { depth: 2 }
  semantic_blocks: { min_tokens: 32, max_tokens: 2048 }
  compile_probes: { sample: 0.005, timeout_s: 30 }
```

## References

- the offline tokenized-enrichment step
- the public data-preparation notes
- the public changelog
- a structure-aware attention plan
- a phase-five ablation plan
- [Tree-sitter - GitHub Tree-sitter project]
- [libclang Python bindings - LLVM project]
- [Aho-Corasick string matching algorithm - Aho and Corasick, CACM 1975]
