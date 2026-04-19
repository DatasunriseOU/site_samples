---
title: "The Clang semantic indexer: translation units, call graphs, and the perf wall"
description: "How the libclang-based semantic indexer feeds `v6_enriched` parquet: compile_commands handling, the per-file translation-unit graph, call and type edges, the failure modes we hit, and the wall-clock cost of ground-truth semantics."
date: "2026-04-18"
tags: ["clang", "data", "indexer", "c++", "pipeline"]
---

The cheap way to extract structure from a C++ corpus is to throw tree-sitter at it and call the resulting AST a graph. That is what lives behind the approximate semantic lanes: fast, embarrassingly parallel, never blocked on a build system. The expensive way is to drive an actual C++ frontend, parse with full semantic analysis, resolve overloads through the libclang AST, and spend an afternoon per repository when nobody is paying attention. That second path produces the clang-resolved semantic lanes and the enriched long-context surface. This post is how that indexer is built, what it costs, and where it falls down.

## What it is

The semantic indexer is a few thousand lines of Python wrapped around the `clang.cindex` libclang bindings. It exists for one reason: tree-sitter knows a token *looks like* a call expression, but cannot tell you which function it resolves to under the project's `-std=`, `-D`, and `-I` flags. Cross-file resolution, overload selection, template instantiation, and qualified-name extraction need a real frontend. We pay for that.

The pipeline per project is straightforward in shape and gnarly in detail:

1. Walk the project directory with a skip list pruned in-place so we never descend into third-party, external, examples, fuzzers, and the rest of the noise directories that bloat repos without teaching the model anything new.
2. For each `.cpp`/`.cc`/`.cxx`/`.c` file under 500 KB, parse the translation unit with libclang, optionally with the file's `compile_commands.json` entry as the compile arguments.
3. Walk the resulting AST and extract function definitions, their qualified names, their source extents, and the qualified names of their callees.
4. Build a project-wide `ProjectIndex` mapping `qualified_name -> FunctionDef`, plus per-file preambles (includes, usings, typedefs, forward decls).
5. Compute reverse edges (callee to callers), then BFS dependency levels from leaves so we can order functions bottom-up in the output document.
6. Emit training documents that look like a real C++ unit: preamble, then types, then transitive callees, then the root function — sorted by `dep_level` so leaves come first.

When enriched mode is on, the same walk fills four parallel char-level arrays — `symbol_ids`, `call_targets`, `type_refs`, `def_use` — and a tree-sitter pass adds `ast_depth`, `sibling_index`, and `ast_node_type`. Those become the dense metadata columns that feed the enriched dataset surface.

## Compile commands: the entire ball game

The single most leveraged input is `compile_commands.json`. Without it, libclang guesses the language standard, misses 30 %+ of cross-file calls, and silently parses headers as C in projects that are actually C++17. With it, callee resolution is dramatically better and we record authoritative `build_info` per document. Two things bit us repeatedly.

The first is shell parsing. The CMake-emitted database often stores the full compiler invocation in a single command field rather than a structured argument array. We were splitting on whitespace, which decapitated quoted definitions and include paths into garbage tokens. The fix is shell-aware splitting. That single change recovered a non-trivial chunk of cross-file edges on representative projects.

The second is sanitizer flags. Projects under sanitizer builds frequently ship `-fsanitize=address`, `-shared-libasan`, `-fsanitize-recover=...` and a handful of related runtime-only flags. These are valid for the compiler driver but make libclang abort or emit nothing useful — they expect a runtime that is not present at index time. We strip them in `_sanitize_compile_args_for_clang` against an explicit deny-list and a small set of prefix patterns. Without that, entire repositories returned zero functions because every TU parse silently failed.

When `compile_commands.json` is missing or returns shell-script garbage (configure-time substitutions left as separator-heavy soup, unresolved environment variables), the sanity check rejects it and we fall back to a minimal syntax-only argument set plus heuristically discovered include directories. Fewer edges, more orphan callees, but honest and it does not crash.

## The translation-unit graph

The working representation is small and on purpose. Each function record carries name, qualified name, source location, text, callees, and dependency level. The project index is a compact map of functions, per-file groupings, and lazily built callers. Edges are not stored as objects; a function's outgoing edges are its `callees` list, and the reverse map is built once before dependency-level computation.

Dep levels come from a BFS from leaves: every function with no in-index callees gets level 0 and is enqueued, then we walk callers and assign `level = max(callee_levels) + 1`. Cycles are handled by giving any remaining unprocessed nodes `max_level + 1` after the BFS drains. The result is a partial order good enough to write functions out so callees appear before callers in the same document, which is what the model wants to read.

Callee extraction is where libclang earns its money. `extract_callees` walks `CALL_EXPR` cursors and resolves each via `cursor.referenced` to a definition; we then take its `get_qualified_name` (a walk up the semantic-parent chain joined by `::`). System and stdlib calls are dropped via a prefix list (`std::`, `boost::`, `__builtin`, `printf`, `pthread_`, `EXPECT_`, `ASSERT_`, `TEST`, ...) so we are not poisoning the call graph with nodes the model cannot do anything useful with. For template-dependent and overloaded calls where `referenced` is `None`, the v5 worker added a small fallback that walks `DECL_REF_EXPR`, `MEMBER_REF_EXPR`, and `OVERLOADED_DECL_REF` children — that recovered roughly 30 % more edges in measurements on the v5 Kubernetes-orchestrated pipeline, and the same fallback is what the bundled indexer ends up using on the same translation units.

## Cross-file edges and the document layout

The semantic indexer does not stop at intra-file edges. After the project-wide index is built and dep levels are computed, `build_training_documents` iterates over every definition, collects the transitive deps via `collect_transitive_deps` up to `max_dep_depth` (5), sorts them by `dep_level`, and assembles the document as `preamble + sorted deps + root function`. If the result blows past `2 * max_tokens` we trim from the highest-level deps first — those are the most peripheral, the model loses the least if we drop them.

Three things to know about that layout. It is deduplicated by md5 over the assembled text — with template-heavy projects (boost, llvm), the same root function pulls in identical transitive deps via different starting points, and dedup is the only thing keeping the corpus from being half copies. It throws away anything under 20 estimated tokens; tiny stub functions add no signal and inflate the document count. It records `dep_level` in the chunk boundaries, which becomes the field consumed by the structure-aware dataloader and turned into a learned `dep_level_emb` at the input. The indexer is not just emitting text — it is emitting graph annotations the model trains against.

For `--enriched`, the same `parts_info` (preamble + deps + root) is replayed to fill `structure_ids` per character, then `extract_semantic_metadata_from_parts` walks the same TU again to paint per-character `symbol_ids`, `call_targets`, `type_refs`, and `def_use`. Symbol IDs are the lower 31 bits of an md5 of the qualified name, with `0` reserved for "no symbol". Reusing md5-of-qname keeps the IDs stable across runs and across the v4/v5/v6 lanes that share the same hashing convention.

## Failure modes we have actually seen

The indexer is deceptively simple but it has crashed in interesting ways. Recording them so we do not re-discover them.

**libclang not found, but `rc=0`.** The bindings dlopen `libclang.so` from a candidate list (`/usr/lib/llvm-21/...`, the bundled `clang/native/libclang.so`, env vars `MEGACPP_LIBCLANG_PATH` / `LIBCLANG_PATH`). Before we added the bundled-first preference and the env-var fast path, jobs stalled behind a system-wide library scan, fell back to a wrong libclang version, then returned `rc=0` with empty output. Even with libclang present, single-project failures used to be swallowed the same way. The orchestrator misclassified all of those as legitimate "lane empty" results. The fix lives in `tools/clang_indexer/index_project.py` and the build3 manager: failures propagate non-zero, and the manager refuses to publish a lane without an explicit empty marker.

**`-x c++` on `.c` files.** The fallback args once unconditionally appended `-x c++`, which forced C++ mode on `.c` files and broke parses. The fallback now leaves language detection to the file extension and only forces C++ when we know the file is C++.

**CDB drift between commits.** In the commit-history producer, `cmake` was only run once at HEAD, so as the worker walked back through commits, `compile_commands.json` was wrong for any commit where `CMakeLists.txt` had since changed. The fix regenerates the CDB whenever the build files change between commits. Without it, `compile_args` were silently wrong and so were the resulting edges.

**Recursion limits on huge ASTs.** `gcc-mirror`, `llvm-project`, and parts of boost have ASTs deep enough to blow Python's default recursion limit. We bump `sys.setrecursionlimit(50000)` at module import. It is ugly and we know it; the alternative is converting every visitor to an explicit stack, which is a lot of code to maintain for a few outlier repos.

## The performance wall

Tree-sitter chunking runs at roughly 1000 files/sec on a normal box. The clang indexer does not. A single TU parse with `PARSE_INCOMPLETE | PARSE_PRECOMPILED_PREAMBLE` plus the AST walk costs us tens to hundreds of milliseconds per file depending on how many headers transitively pull in, and the dominant share of wall time is libclang itself, not Python.

We have tried four things to push that wall back. `PARSE_PRECOMPILED_PREAMBLE` reuses the parsed preamble across reparses of the same file. `PARSE_INCOMPLETE` lets libclang return what it has even when the TU does not fully type-check, which keeps the indexer alive on projects whose checked-in headers expect a configure step we did not run. `ProcessPoolExecutor` instead of threads is non-negotiable: threads cannot be killed when libclang hangs on a pathological file, and once you have one stuck thread per worker, the entire pool degrades — the v5 worker rewrite documented "55 of 55 futures unfinished" on 128 K shards before the switch. Aggressive directory-level skip lists (no the public test suite, `examples/`, `fuzzers/`, `third_party/`, `external/`, `vendor/`, `build/`) roughly halve wall time on `llvm-project` and `boost`. A 500 KB per-file ceiling drops the generated outliers.

Even with all four, the semantic lane is still the slowest producer we run. The yardstick is the relative output sizes on the reference corpus: one clang-resolved 16K lane is **794,028** docs at **883 MB** of parquet, while the cheaper tree-sitter 16K lane on the same source produces **8,092,541** docs at **23 GB**. The clang lane emits roughly **10 %** of the documents and **4 %** of the bytes for the same input. We accept that ratio because the documents we do emit are the only ones with cross-file edges resolved through a real C++ frontend.

## What we use it for

Two consumer surfaces depend on this indexer.

The first is the production semantic training shard, mixed at 0.6 against a commit-history shard at 0.4 in the current launchers. That dataset is what the long-context specialists learn cross-file repository reasoning from. It has authoritative call edges, deduplicated bottom-up assembly, and `dep_level` annotations that drive the structure-aware curriculum.

The second is the enriched dataset family. The semantic indexer's enriched mode is what fills the symbol-ID and type-ref char arrays that the structure-aware loader feeds into the model as input embeddings. The structure-aware design is only meaningful because somebody bothered to compute the per-character categories from a real semantic walk rather than from regex.

## What we are not doing, and what is left

We do not run the full indexer in CI. The cost is too high and the inputs (raw repositories, libclang version, OS-level headers) vary enough that a green check would not mean much. Instead, we run focused regression coverage on the deterministic parts of the pipeline and pin the libclang version per build image. We also do not attempt whole-program LTO-style analysis: the indexer is project-local, cross-project edges are not resolved, and the training corpus carries those projects as separate but co-visible sources.

Two open items. A streaming version of the indexer that does not require holding the entire `ProjectIndex` in memory; on `linux v6.10` and `pytorch`, peak RSS runs into multiple GB. And a better story for templates: we dedup by md5 of the assembled text, but template-heavy projects emit many slightly different documents that are semantically equivalent for the model. Some normalization pass — strip template parameters, hash on structure rather than text — would probably cut the dataset 10–20 % at no quality loss. We have not measured.

The honest summary is that the clang semantic indexer is the most expensive piece of our data plumbing, gives us the highest-quality slice of the corpus, and is the producer we trust least to be running healthy without explicit per-lane validation. Both halves of that statement are accurate, and both are why we built the rest of the pipeline (consumer-side tolerant loader, multi-lane producer manager, atomic finalize) the way we did.

## Index snapshot

| Artifact | Producer | Consumer |
|----------|----------|----------|
| Translation-unit index | libclang AST walk | build-context feature |
| Macro expansion map | preprocessor hook | dedup, packing |
| Include graph | driver shim | license filter, build-context |
| Symbol -> chunk index | post-walk reducer | training loader, eval verifier |

```text
Minimal semantic-index invocation shape:
- frontend: clang-compatible parser
- mode: syntax-only AST dump
- language standard: project-specific
- include roots: project-specific
- output: structured index artifact
```

## References

- [Semantic indexing notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/semantic-indexing-notes.md)
- [Data preparation notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/data-prep-notes.md)
- [Reference corpus pins](https://github.com/DatasunriseOU/site_samples/blob/main/docs/reference-corpus-pins.md)
- [Sample `compile_commands.json`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/compile_commands.sample.json)
- [JSON Compilation Database Format Specification](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
- [Libclang tutorial and API index](https://clang.llvm.org/docs/LibClang.html)
