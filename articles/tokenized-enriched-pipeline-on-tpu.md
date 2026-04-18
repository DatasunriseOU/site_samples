---
title: "Tokenized Enriched Packed Rows on TPU: Feeding Structure to XLA Without Recompiles"
description: "How the v6_enriched packed-rows pipeline feeds per-token structure_ids, chunk_boundaries, and call_edges into the XLA dataloader on TPU v6e without triggering compile cache misses, and how the contract lifts into the main path."
date: "2026-04-18"
tags: ["tpu", "xla", "data-pipeline", "structure-aware", "packed-rows"]
---

The point of an enriched dataset is to teach a transformer that code has structure that is not in the bytes: function boundaries, dependency levels, call edges, type references. The point of a packed-row layout is to keep the long-context step at maximum tokens-per-second by stuffing many short documents and a few large ones into a fixed-shape window. Doing both on TPU means letting an MXU-bound model see structure-aware data without the XLA compile cache exploding every batch. This post is how we useped that pipeline on the v6_enriched parquet format, what the dataloader has to canonicalise to keep the compiled graph stable, and how the contract lifts into the deployment builder.

## Why MegaCpp cares about this

Pretraining on raw C++ tokens leaves a lot on the table. The byte-level model has to relearn that `class`, `};`, and indented function bodies mean something; that `#include` hints at which symbols to expect; that the call graph beats lexical proximity. Our enriched parquet turns that into per-token data: `token_structure_ids` (9 categories), `token_dep_levels` (BFS depth in the dep graph), `token_ast_depth`, `token_sibling_index`, `token_ast_node_type`, plus chunk-level `token_chunk_starts`/`ends`/`kinds`/`dep_levels`, plus `token_call_edges` and `token_type_edges` between chunks. On TPU v6e you cannot pay for this with host-side recompilation; every Python branch on column presence is a graph cache miss.

The win, when the pipeline holds, is twofold. First, a meaningful throughput recovery from switching to pretokenized columns over the legacy char-level format (~4x in the dataloader microbenchmark: roughly 17,000 vs 4,000 tok/sec). Second, the structure embedding and TreeFFN paths become cheap on TPU once the chunk mapping is precomputed at packing time and threaded through as static-shape tensors.

## What we built in the POC

Four modules carry the contract: the public tokenized-enriched schema sample, the public tokenized-enriched pipeline sample, the public packed-rows schema sample, the public structure-embeddings sample. The schema modules hold column names and coercion helpers; the materializer holds the offline char-to-token conversion; the structure module is the model-side consumer.

the public tokenized-enriched schema sample is the source of truth for column names: `TOKEN_IDS_COLUMN`, `TOKEN_STRUCTURE_IDS_COLUMN`, `TOKEN_DEP_LEVELS_COLUMN`, `TOKEN_AST_DEPTH_COLUMN`, `TOKEN_SIBLING_INDEX_COLUMN`, `TOKEN_AST_NODE_TYPE_COLUMN`, `TOKEN_SYMBOL_IDS_COLUMN`, `TOKEN_CALL_TARGETS_COLUMN`, `TOKEN_TYPE_REFS_COLUMN`, `TOKEN_DEF_USE_COLUMN`, plus chunk-level `TOKEN_CHUNK_STARTS/ENDS/KINDS/DEP_LEVELS_COLUMN` and `TOKEN_CALL_EDGES_COLUMN` / `TOKEN_TYPE_EDGES_COLUMN`. TypedDicts and predicates (`_is_token_value_sequence`, `_extract_span_bounds`) keep producer and consumer agreeing on shapes.

the public tokenized-enriched pipeline sample is the offline materializer. `materialize_tokenized_enriched_batch(docs, tokenizer)` encodes texts with `encode_batch` to recover per-token character spans, then walks each char-level metadata array assigning each token the value of its first character (`_chars_to_tokens_structure_ids`). Chunk boundaries become token offsets (`_chunk_boundaries_to_token_offsets`); per-token dep levels come from chunk-level dep levels and the token-to-chunk mapping (`_compute_token_dep_levels`); edges are remapped from char-level to token-level chunk indices (`_remap_token_edges`). The materializer refuses to proceed if the tokenizer does not produce per-token character spans; silently emitting unaligned metadata would be a correctness bug invisible until the model misbehaved.

the public packed-rows schema sample defines the runtime contract. Packed rows have a fixed layout: `pack_id`, `input_ids`, `target_ids`, `loss_mask`, `doc_ids`, `valid_token_count`, `num_docs`, plus optional provenance. `PACKED_ROWS_TOKEN_ALIGNED_COLUMNS` and `PACKED_ROWS_CHUNK_METADATA_COLUMNS` are explicit tuples so consumers never guess. `PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES` splits fills deliberately: zero for category-style columns (structure, dep level, change masks), `-1` for true sentinels (ast depth, sibling index, ast node type, hunk id). The dataloader uses these to keep batch shapes stable.

the public structure-embeddings sample is the model-side consumer. The TPU fast path in `StructureGraphEnricher.forward` exists because TPU is allergic to scatter, searchsorted, topk, and nonzero: each lowers to host-syncing or shape-fragile ops. The TPU path replaces them with bmm-based pooling and cumsum-based top-K neighbour selection.

## Feeding structure_ids, chunk_boundaries, call_edges through XLA without recompiles

Three sources of recompilation bite enriched data: variable presence of optional metadata, variable shapes per batch, and Python branches inside the model that read tensor values to choose a path. We addressed each.

The first is solved at the boundary, in `_canonicalize_structure_meta_for_xla(x, structure_meta)` in the TPU training launcher. On the XLA branch this helper materialises every optional enriched tensor at a stable shape when the corresponding feature is enabled. With structure embeddings on, `token_structure_ids`, `token_dep_levels`, `token_ast_depth`, `token_sibling_index`, `token_ast_node_type` are present as `(B, T)` every batch; missing columns get the fallback fills. With TreeFFN on, `token_chunk_ids`, `token_chunk_valid`, `semantic_block_starts/ends/valid`, and the `semantic_block_keep_mask` / `semantic_block_edge_weights` matrices materialise at fixed `max_semantic_blocks = max(128, T // 32)`. Optional tensors the model does not need are popped explicitly so the dict has the same key set every batch.

The second is solved at packing time. Chunk boundaries vary per row, so the dataloader pads them to `FIXED_MAX_CHUNKS = max(128, T // 32)` (matching the canonicaliser) with zero starts/ends. Valid-chunk counts are recoverable from `ends > starts`. `chunk_relation_mask` (call/type edges as a per-chunk relation matrix) is padded to `(B, R, FIXED_MAX_CHUNKS, FIXED_MAX_CHUNKS)`. Padding is the cheapest fix; the alternative is per-batch dynamic shapes the partitioner cannot cache.

The third is solved inside the model. `StructureGraphEnricher.forward` keys on `_use_bmm = token_chunk_ids is not None or x.device.type == "xla"`, so the XLA path always uses bmm pooling. When the dataloader supplies `token_chunk_ids` / `token_chunk_valid` (the common case after wiring) we skip the searchsorted reconstruction; otherwise we fall back to a sort-then-searchsorted path that is still XLA-clean (sort over a sentinel-padded array, gather to invert the permutation) but slower. The TPU bmm pooling reads `F.one_hot(chunk_ids, C).permute(0, 2, 1)` masked by `valid`, then `bmm(membership, x)` and normalises; it lands on the MXU.

Neighbour selection is similarly XLA-shaped. CUDA uses `topk(adj, K, dim=-1)`; on TPU `topk` lowers to a full sort plus slice on the VPU, expensive and shape-fragile. The XLA path uses `cumsum` over the boolean adjacency to compute a 1-based rank, masks `cumsum <= K`, then `argsort(descending=True, stable=True)[..., :K]`. Same result, MXU-friendly, deterministic. The neighbour scatter for incoming aggregation becomes `bmm(neighbor_membership_t, msg_flat)` instead of `scatter_add`.

## Varlen handling

Packed rows mix many small documents with a few large ones. The model sees a single `(B, T)` tensor with `doc_ids` and `valid_token_count`; the varlen contract lives at three layers. At the parquet layer, the packer guarantees `doc_ids` is a monotonically increasing per-token int array within a row and `valid_token_count` is the prefix length. `PACKED_ROWS_PACKER_REQUIRED_COLUMNS` enforces it; a row missing any required column is rejected at load time, not zero-filled.

At the attention layer, dense paths see `doc_ids` and build a per-token same-doc causal mask. The CUDA varlen FA path derives `cu_seqlens` from `doc_ids` (`flash_attention.get_cu_seqlens_from_doc_ids`) outside the compiled region; the cu_seqlens tensor itself is small. On TPU the Pallas FA kernel takes `segment_ids` derived from `doc_ids` once at the boundary.

At the structure layer, `valid_token_count` drives `normalize_attention_validity` to produce an `AttentionValidity` carrier. The blockized sparse path consumes this via `classify_selected_block_masks`; the structure embedding consumes it implicitly via `token_chunk_valid`. Neither infers validity inside the compiled region. The kernels run on dense `(B, T)` tensors with masks from precomputed metadata; the partitioner caches one graph per recipe.

## How it lands in deployment

The deployment builder ships the same contract with a Megatron-shaped consumer. the public embedding sample ports the additive structure embedding as `CppMegaStructureEmbedding`; the public config sample carries the fail-closed `StructureConfig` that translates POC argparse flags into a frozen dataclass. The component allowlist is `("structure", "dep_level", "ast_depth", "sibling_index", "ast_node_type")`; deployment default is `"core"` (`structure` + `dep_level`). Bottleneck dim defaults to 64, matching the POC ablation.

the public custom-embedding sample subclasses Megatron's `LanguageModelEmbedding` and adds the structure embedding (and the n-gram hash embedding) as additive contributions before the Megatron forward. The sharded-state-dict walker is patched so custom submodules get distinct `replica_id` stamps when MTP replicates the embedding on a non-first pipeline stage; without that fix, the default walker stamps a duplicate "main replica" and the checkpoint becomes ambiguous.

the public structure-batch sample is the small bridge that extracts structure inputs from a batch dict (`structure_ids`, `dep_levels`, `ast_depth_ids`, `sibling_index_ids`, `node_type_ids`) and threads them onto the model via `set_structure_inputs`. Conditional setting is fine here; the equivalent of `_canonicalize_structure_meta_for_xla` on the TPU deployment path is a recipe-level guarantee that the loader emits the columns every batch.

Lifted as-is: schema column names, dense fallback fills, the varlen contract on `doc_ids` + `valid_token_count`, the `AttentionValidity` normalisation, bmm-based pooling and cumsum-based neighbour selection. Rewritten: the offline materializer becomes a Megatron preprocessing job; packing moves to the deployment data pipeline; the TreeFFN loop is recipe-flagged and defaults off pending more ablation at deployment scale. Dropped: the legacy char-level `enriched_code_v3` format does not enter the main path; we migrated to pretokenized `v6_enriched_*` because the char-to-token conversion was the dataloader bottleneck. Feature-flagged: `tree_ffn_enabled`, `relation_bias_enabled`, `platform_embed_enabled` stay as recipe flags; the additive structure embedding is the default-on minimum.

## Ablations and what we kept

The wins that survived the migration into the deployment loader:

| Change | Where | Effect |
|--------|-------|--------|
| Pretokenized over char-level enriched | the public tokenized-enriched pipeline sample | ~4,000 -> ~17,000 tok/sec dataloader |
| `F.conv1d` regardless of `doc_ids` | mixer | ~22% recovered on CUDA |
| Bottleneck dim 64 | the public structure-embeddings sample | -23% throughput, kept for quality |
| Precomputed `token_chunk_ids/valid` | `GPT.forward` | Skips `searchsorted` in TreeFFN |
| `_canonicalize_structure_meta_for_xla` at boundary | the TPU training launcher | Eliminates per-batch graph cache misses |

From the CHANGELOG entries that anchor each decision:

- **Pretokenized over char-level.** Raw char-level enriched ran around 4,000 tok/sec in the dataloader; the pretokenized format gives roughly 17,000 tok/sec. The materializer in the public tokenized-enriched pipeline sample is what makes pretokenized authoritative.
- **Conv1d on CUDA regardless of `doc_ids`.** The 4-iteration manual depthwise conv was triggering whenever `doc_ids is not None` (always true on enriched data). Switching back to `F.conv1d` recovered ~22%; cross-doc leakage at most 3 tokens for kernel=4 is negligible.
- **Bottleneck dim 64.** Initial cost was around 23% throughput, but the absolute number stayed competitive once paired with the precomputed chunk mapping. Production keeps 64.
- **Precomputed `token_chunk_ids` and `token_chunk_valid`.** `GPT.forward` threads them into `StructureGraphEnricher`, so TreeFFN skips the searchsorted reconstruction. The fast path was wired through CUDA TP broadcast, CP sharding, and XLA/CUDA structure-meta canonicalisation in the same window.
- **Canonicalisation at the XLA boundary.** Per-batch presence-conditional Python branches inside the model produced cache misses; pulling canonicalisation to the train script and materialising every optional tensor at fixed shape eliminated them.
- **Bench harness.** the sanitized enriched-loader benchmark emits a JSON report with B, T, iters, token rate; re-run on every dataloader change.
- **Validator coverage.** 45 tests gate the packed-row schema across sanitized packed-row schema tests, sanitized packed-row metadata tests, sanitized dataloader packed-row tests, sanitized H200 sparse validation smoke tests. They have caught at least two regressions where the producer dropped a token-aligned column and the consumer densified to garbage.
- **Schema split.** Producer-local provenance fields stay in the offline packer until producer and runtime contracts unify; the runtime schema only depends on what the dataloader actually consumes.

## Production checklist

The boundary canonicaliser is the load-bearing line on the XLA path:

```python
# the TPU training launcher — XLA branch only
def _canonicalize_structure_meta_for_xla(x, structure_meta):
    # Materialise every optional enriched tensor at fixed shape.
    # Missing columns get PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES.
    structure_meta = _ensure_token_aligned(structure_meta, B, T)
    if tree_ffn_enabled:
        structure_meta = _ensure_chunk_metadata(
            structure_meta, max_chunks=max(128, T // 32),
        )
    return structure_meta  # shape-stable; no Python branches downstream
```

- Lock the parquet schema per recipe; reject loads where `PACKED_ROWS_PACKER_REQUIRED_COLUMNS` are missing.
- Run `_canonicalize_structure_meta_for_xla` at the train-script boundary on the TPU path; no presence-conditional Python branches inside the model.
- Pin `FIXED_MAX_CHUNKS = max(128, T // 32)` between dataloader, canonicaliser, and model; bump in lockstep.
- Default structure embedding to `"core"` with bottleneck dim 64; gate `ast_depth`, `sibling_index`, `ast_node_type`, TreeFFN behind explicit recipe flags.
- Keep the precomputed `token_chunk_ids` / `token_chunk_valid` path on; reject data drops that lose these columns.
- Use bmm-based pooling and cumsum-based neighbour selection on TPU; no `topk`, `scatter_add`, `searchsorted`, or `nonzero` inside the compiled region.
- Derive `cu_seqlens` (CUDA) and `segment_ids` (TPU) from `doc_ids` outside the compiled region; cache per batch.
- Run the sanitized enriched-loader benchmark on every data-side change; gate landing on a tok/sec regression budget.
- Refuse the legacy char-level enriched format; force the pretokenized path.
- Validate per-token character spans before the offline materializer runs; do not silently emit unaligned metadata.

## References

- the public tokenized-enriched pipeline sample, the public tokenized-enriched schema sample, the public packed-rows schema sample, the public structure-embeddings sample
- the public dataloader sample (FIXED_MAX_CHUNKS handling), the TPU training launcher (`_canonicalize_structure_meta_for_xla`)
- the public embedding sample, the public config sample, the public custom-embedding sample, the public structure-batch sample
- the engineering changelog entries on the pretokenized format switch, the precomputed chunk mapping wiring, the structure-emb bottleneck dim, and the XLA structure-meta canonicalisation.
- [TreeGPT: Treeified Code Modeling — paper reference]
- [PyTorch/XLA SPMD documentation — official `torch_xla` docs]
- [Pallas: A JAX Kernel Language — official JAX docs]
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs — Xu et al., 2021]
