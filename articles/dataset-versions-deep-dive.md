---
title: "Dataset Versions v2 to v6: The Long-Form Ablation History"
description: "A detailed walk through every schema generation of the C++ training corpus - what each version added, the schema diff, the storage cost, the val_bpb delta we attribute to each step, what we deprecated and why."
date: "2026-04-18"
tags: ["data", "dataset", "ablation", "schema", "history"]
---

The high-level overview of v2 through v6 lives in a separate post; this one is the long-form ablation history for engineers about to add a `v7`. The discipline that has saved us the most pain is that no version replaced the previous - they coexist in the object store and load through the same tolerant consumer.

## Why MegaCpp cares about this

Migrating a corpus across schema boundaries is the most expensive thing you can do in a data stack. Re-tokenizing 27.6 M documents because you renamed a column burns a week of compute time. Each generation was paid for in large compute budgets and double-digit terabytes; writing the history down is how the next version avoids the same mistakes.

## What we built in the POC

The shared substrate before any version: eight pinned C/C++ repositories cloned shallow at explicit refs, totaling ~15 GB on disk after shallow clone, plus a 142-repo catalog tracked separately. One hybrid C++ tokenizer (now 131,072 entries). The differences between versions are how we *chunk*, *order*, *enrich*, and *resolve cross-file references* over those same bytes.

**v2 - full files, pre and post commit.** The first version that was more than a flat dump. Walks each repo's commit history and, for each commit touching a C++ file, emits the file before and after as two documents. The point is temporal signal: flat files have no notion of code as something that used to look different.

Schema: JSONL with `{"text", "repo", "commit_hash", "filepath", "timestamp"}`. No structure metadata. Tokenized to `uint16` binaries because the early tokenizer fit in 16 bits. Storage: roughly a couple of TB of compressed JSONL across 27.6 M documents.

Val_bpb attribution: v2 is the baseline against which every later version is measured operationally, not statistically. Phase 1 of the curriculum (4K context, syntax mastery) lives entirely on v2/v3 packed into 4096-token sequences. We never ran a clean v2-vs-flat-dump ablation at scale because the flat-dump baseline was already gone by the time we had the infrastructure to do it.

**v3 - structured inline diffs.** Same commit walk, different rendering. Each commit becomes a single document under a synthesized file header, with removed lines as C++ comments (`// Removed: ...`) and added lines as live code. Two header styles ship side by side: `v3_doxygen` (Javadoc) and `v3_simple` (plain `// File: ...`).

Schema diff vs v2: same JSONL envelope; `text` is now a synthesized commit document. New optional `header_style` field. Still `uint16`. Storage roughly doubles to carry both header styles.

Why both styles: we did not know which the model would learn faster from. Doxygen matches the conventions of LLVM and folly; simple is less noisy. Rather than guess, we built both, packed them at equal weight, and let evals speak. They did not differ enough to justify keeping only one, and once both exist the marginal cost is storage.

Val_bpb attribution: v3 teaches "what was changed and what it replaced" instead of v2's "what does this file look like before and after." Cleanly attributing a BPB delta to v3-vs-v2 alone is hard because the curriculum phases that consumed v3 also enabled FIM, document masking, and tokenizer changes simultaneously. Operational evidence: v3 unblocked diff-shape learning and we kept it.

**v4 - tree-sitter context graph.** The first version that emits a graph. For every modified function in a commit, build a strict 64K-token window containing the target plus its direct callers and direct callees, extracted with a tree-sitter AST walker.

Schema diff vs v3: still text-shaped at the consumer surface, but `text` is now a graph-assembled window. New optional `language_info` (header reclassification, CUDA/HIP/OpenCL markers, SQL primary gating, C++-standard hints), `platform_info` from the platform scanner described in the public data-preparation notes, and `build_info` for compile-command provenance.

Storage cost: significant. Shards live in size buckets (4K/16K/64K/128K) that share schema lineage with v4. A single double-digit-TB local cleanup pass deleted four verified `v7` tree-sitter waves once they were verified complete on object storage.

Operational history: v4 had an embarrassing first month. A JSON schema deserialization bug in the Rust binary produced zero-byte outputs for a non-trivial fraction of repos. The pipeline ran healthy at 100% CPU for hours and emitted plausible shard counts. The empty-document round-trip gate caught it. Once fixed, the pipeline saturated 40 cores and hundreds of repos finished cleanly.

v4 is approximate by design. Tree-sitter does not resolve names across files, does not see overloads, does not know which `foo()` a `call_expression` resolves to under namespaces or templates. For 16K curriculum windows that is fine; the point of v4 is throughput.

Val_bpb attribution: v4 is what made Phase 2 (file-level reasoning, 16K context) work. Phase 1 ablation shows DSA on v3-shaped data at val_bpb ~1.562 (the largest single-feature improvement, -16.3% over the attn-only baseline of ~1.866). We do not have a clean "DSA on v4 vs DSA on v3" number at fixed model config because the context-length change confounds it.

**v5 - libclang semantic graph.** The answer to "what does v4 lie about." Where tree-sitter approximates, libclang resolves. v5 drives Clang with each project's `compile_commands.json`, walks git history incrementally, and emits 100%-accurate semantic relationships: cross-file calls resolved through the actual frontend, types through the actual symbol table.

Schema diff vs v4: same envelope; graph content is now semantically resolved instead of AST-approximated. `build_info` becomes authoritative per record. `language_info` is preserved when uniform across constituent files.

Storage cost: tens of GB across the 4K/8K/16K/32K/64K/128K buckets. The producer is far more expensive than v4 - per-commit incremental builds are slow, and a non-trivial fraction of repos cannot be built reproducibly at all (embedded RTOSes we cannot mirror).

Operational history: deployment took longer to stabilize than the indexer itself. We hit image-pull failures until the worker service account got artifact-registry read access, `ThreadPoolExecutor` deadlocks on hung clang TUs (fixed by switching to `ProcessPoolExecutor` with a per-commit lifecycle), and Chromium's 51 GB working tree stalling under `git checkout --force` until we rewrote extraction as `git show commit:filepath` over a `git diff-tree` change list. The worker pool scaled from 50 to 246 pods across size buckets.

Val_bpb attribution: v5 is the only producer in the stack whose call edges we trust at long context. Tree-sitter's are wrong often enough at 64K that we cannot use them as the sole source for repository-reasoning training. Phase 3 (64K, repository graph reasoning) does not exist without v5.

**v6 - enriched parquet.** The version where the dataset stops being just text. Same commit walk. Same v5-quality semantic edges where available, v4 tree-sitter edges as fallback. The change is the schema: each parquet record now carries dense structural metadata as additional columns.

Schema diff vs v5: parquet instead of JSONL. New columns include `structure_ids` (`list<u8>` keyed to one of nine categories: `other`, `preamble`, `func_sig`, `func_body`, `class_decl`, `class_member`, `comment`, `typedef`, `namespace`), `chunk_boundaries`, `call_edges`/`type_edges` indexing those boundaries, optional per-character AST metadata (`ast_depth`, `sibling_index`, `ast_node_type`), and the preserved `platform_info`/`language_info`/`build_info` triple. A token-level extension materialized offline (`token_ids`, `token_structure_ids`, `token_dep_levels`, plus chunk-level mirrors and edges) is what the production loader actually consumes.

Storage cost: enriched parquet is roughly comparable to v5 JSONL compressed - the new columns are sparse and parquet handles them well. The token-level extension roughly doubles per-record size for the documents we materialize, which we do selectively.

The tokenizer also jumped under v6: 131,072 entries, exceeding `uint16`. Pretokenized v6 shards switched to `uint32`. Older v2/v3 `uint16` archives remain valid against the older tokenizer; the new ones are not.

Val_bpb attribution: this is where we have the cleanest receipts. The R6 ablation matrix from March 2026 measured a depth-52 hybrid 4.1B preset on the same hardware with the same step count, varying only the feature stack:

- R6-E (bare baseline): 595,585 tok/sec, 2.493 BPB
- R6-A (DSA + Engram + mHC, no enriched data): 496,962 tok/sec, 2.299 BPB
- R6-F (full: enriched + structure_emb + tree_ffn + DSA + Engram + mHC): 316,047 tok/sec, **2.287 BPB**

The enriched-plus-structure marginal effect over R6-A is small in BPB (-0.012) at significant throughput cost (-36% relative). It stays because the BPB win compounds across the curriculum and the throughput cost is recoverable: pretokenized columns gave 4x in the dataloader microbenchmark, eager segment materialization gave 2.4x, conv1d-on-CUDA-regardless-of-doc_ids gave 22%, and `structure_bottleneck_dim=64` gave 23%. Post-fix, the accelerator-backed kernel breakdown shows enriched features within noise of the dense baseline.

## How it lands in MegaCpp

MegaCpp ships v6 enriched parquet as the primary training surface. v2/v3 archives stay on storage as cheap reproducibility insurance for old checkpoints. v4 outputs with a v5 replacement get garbage-collected only after the remote copy verifies complete and the consumer no longer reads the local copy. v5 outputs are sacred until a revisioned successor under the same schema publishes. v6 is what every new ablation runs against.

Producer-side rewrites: the legacy flat-text producer is sunset; only the strict producer with exact-token budgeting and pretokenized columns ships. The `uint16` binary path is dropped. The `RelationBiasComputer` consumer path is dropped from all presets, with the edges remaining in parquet for the TreeFFN consumer.

Schema generation and producer revision counter stay on different axes: a producer revision is still v5-schema; a hypothetical v7 would coexist with v6 the way v6 coexists with v5. We do not migrate unless the migration cost is provably lower than parallel-storage cost.

## Ablations and what we kept

The ablation history that mattered most was operational rather than statistical. Phase 1 showed DSA at val_bpb ~1.562 versus a baseline of ~1.866 (-16.3%) - the largest single-feature improvement we have recorded. Phase 2 added Engram and mHC over DSA; both helped, MTP slightly hurt at 4K. Phase 3 validated 128K as the practical context maximum on a TPU v6e slice (256K OOMs under XLA pre-allocation even with gradient checkpointing). Phase 5 stacking produced an MoE preset at val_bpb 1.206, substantially better than the best dense result at the same step.

Data-version-specific ablations that survived contact with real GPUs:

- v6 char-level vs v6 pretokenized: 4.1x loader throughput, identical model. Kept pretokenized as the only production path.
- v6 enriched vs v6 plain text: small BPB win, large throughput cost recoverable through the engineering above. Kept enriched.
- `RelationBiasComputer` on top of `structure_emb` + `tree_ffn`: removed across all presets. ~154 lines of class, ~700 lines of tests, no measurable BPB at scale.
- Two v3 header styles (`doxygen` vs `simple`): no meaningful difference; both kept because storage is cheap.
- v4 vs v5 collapse attempt: tried in design, cannot. v4 is a throughput layer; v5 is a correctness layer. Different phases, different characteristics. Every collapse reinvented the same split under different names.

Shipping but no clean ablation: `ast_depth` / `sibling_index` / `ast_node_type` under v6. They train stably; we have no proof they add BPB independently of `structure_ids` and `dep_levels`. Cost is small and removing them later is a one-line change, so they stay opt-out.

## Production checklist

- Schema generation (`v2`-`v6`) and producer revision counter are different axes; keep them separate in launcher configs and writeups.
- `text` is mandatory and byte-identical across all v6 records. Everything else is optional.
- `uint32` token width at 131K vocab; the verify gate must reject any `uint16` v6-era pretokenized output.
- Do not delete a producer output unless the remote copy verifies complete, the file count matches, and no consumer still reads the local copy.
- v5 outputs stay until a revisioned successor under the same schema publishes; v2/v3 archives stay indefinitely as reproducibility insurance.
- The empty-document round-trip gate stays on. v4's first month is why.
- The training loader is the source of truth for record shape. Producer-side schema disagreements get resolved by the tolerant loader, not by mass re-emission.

## What we got wrong, recorded

Conflating schema generation with producer revision counter cost real onboarding hours before it became policy.

Letting producers coexist for months turned every "which producer emits what" question into a cross-reference exercise. Sunset older producers deliberately once migration completes.

`RelationBiasComputer` shipped on promising synthetic benchmarks; the promise did not survive real corpora. Directive: do not re-add a learned per-head pair bias on top of TreeFFN without an ablation that beats the cost.

## Version map

| Version | What it added | Schema delta vs prior | Storage cost | Use case |
|---|---|---|---|---|
| v2 | full pre/post commit files | new JSONL envelope | ~few TB | Phase 1 syntax |
| v3 | structured inline diffs | `header_style` field | ~2x v2 | Phase 1 diff shape |
| v4 | tree-sitter context graph | `language_info`, `platform_info`, `build_info` | double-digit TB | Phase 2 16K |
| v5 | clang-resolved cross-refs | typed call/use edges | comparable to v4 | Phase 3 64K |
| v6 | enriched parquet + structure ids | `structure_ids`, `loss_mask`, sidecar provenance | +20-30% over v5 | Phase 4 structure-aware |

```python
# Tolerant loader contract: trust optional columns only when shape-valid.
def load_row(row, T):
    ids = row["input_ids"]
    assert ids.shape[-1] == T
    doc_ids = row.get("doc_ids")
    if doc_ids is None or doc_ids.shape != ids.shape:
        doc_ids = infer_from_bos(ids)
    return ids, doc_ids, row.get("loss_mask")
```

## References

- the public curriculum-mapping notes
- the public changelog
- the offline tokenized-enrichment step
- the public data-preparation notes
