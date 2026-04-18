---
title: "C++ Data Versioning and Schema: Keeping Packed Rows and Loaders Compatible"
description: "How the POC evolved packed rows, tokenized enriched columns, structure IDs, and relation edges without breaking loaders, and which schema discipline now carries into MegaCpp."
date: "2026-04-18"
tags: ["data", "schema", "packed-rows", "structure-aware", "c++"]
---

TL;DR: the POC stopped treating dataset format changes as one-off migrations and started treating them as a compatibility contract. the offline tokenized-enrichment step, the tokenized-enriched schema layer, the packed-row schema layer, the structure-embedding layer, and the relation-bias layer now define a narrow rule: add columns and semantics carefully, canonicalize missing data deterministically, and keep model consumers stable even while the corpus gets richer. That discipline is what lets MegaCpp ship richer C++ structure signals without turning every loader change into a full recook.

## Why MegaCpp cares

The expensive part of a C++ data stack is not tokenization by itself. The expensive part is invalidating everything downstream of tokenization because one metadata field changed shape, one enum gained a new member, or one loader started assuming a column is always present. If you have packed rows on disk, token-level enriched parquet upstream, and multiple consumers that only need subsets of the metadata, schema drift can quietly break three different places at once.

That is why the POC converged on an explicit compatibility model instead of a best-effort one. Token-level enriched files can carry more than raw token IDs: structure classes, AST depth, sibling index, chunk boundaries, call edges, type edges, symbol IDs, dependency levels, and platform signals. Packed rows then flatten that information into a batchable form for the trainer. The model finally consumes a reduced, canonical set of tensors. Each layer has its own job, and none of them are allowed to guess what a missing field means.

The result is boring in the best sense. A loader written against an older packed-row contract can keep running as new enriched columns appear upstream. A newer trainer can accept older rows by filling missing structure tensors with deterministic defaults. That is the difference between shipping a new structure-aware corpus as an additive improvement and triggering a week-long full-stack migration.

The discipline also improved engineering speed. Once the team stopped debating whether every new field required a format reset, they could add structure channels in smaller steps: first materialize them in enriched parquet, then decide whether they belonged in packed rows, then decide whether any model path actually consumed them. That sequencing kept compatibility work local. Schema files changed first, packers changed second, and model consumers changed only when a feature had a real training justification.

## What we built in the POC

The schema boundary starts in the tokenized-enriched schema layer. That layer is intentionally plain: it names the columns and the expected semantic families. Token IDs live in `TOKEN_IDS_COLUMN`. Structure-aware token metadata has its own explicit names such as `TOKEN_STRUCTURE_IDS_COLUMN`, `TOKEN_DEP_LEVELS_COLUMN`, `TOKEN_AST_DEPTH_COLUMN`, `TOKEN_SIBLING_INDEX_COLUMN`, and `TOKEN_AST_NODE_TYPE_COLUMN`. Graph-like chunk metadata is separated again into `TOKEN_CHUNK_STARTS_COLUMN`, `TOKEN_CHUNK_ENDS_COLUMN`, `TOKEN_CHUNK_KINDS_COLUMN`, `TOKEN_CHUNK_DEP_LEVELS_COLUMN`, `TOKEN_CALL_EDGES_COLUMN`, and `TOKEN_TYPE_EDGES_COLUMN`.

That split matters because not every consumer wants the same level of structure. Some only need token IDs and masks. Some need token-aligned structural channels. Some need chunk-level relation graphs. The schema file does not try to collapse those into one opaque payload.

the offline tokenized-enrichment step is where char-level enrichment becomes token-aligned data. The core move is simple but strict: token metadata is derived from token character spans, not from approximate token counts or heuristic slicing. `_encode_batch_with_optional_char_spans(...)` prefers batch tokenization that returns offsets; `_extract_token_char_starts_and_valid(...)` and `_chars_to_tokens_structure_ids(...)` map char-level labels to per-token labels; `_chunk_boundaries_to_token_offsets(...)` remaps chunk boundaries into token offsets; `_normalize_graph_edge_pairs(...)` normalizes graph edge encodings before they move further down the stack.

That gives the schema a clean layering model:

| Layer | Primary artifact | Contract |
|---|---|---|
| Enrichment producer | tokenized enriched parquet | optional, additive columns with explicit names |
| Packing layer | packed rows | fixed row layout plus stable fallback fills |
| Model input | structure meta tensors | dense canonical tensors with stable shapes |

The packing contract is defined in the packed-row schema layer. This is where the POC made the most important compatibility decision: token-aligned columns and chunk metadata columns are enumerated explicitly instead of being discovered dynamically. The schema groups token-aligned fields together and separates chunk-level graph metadata from row-core fields such as `input_ids`, `target_ids`, `loss_mask`, `doc_ids`, `valid_token_count`, and `num_docs`.

The other important decision is fallback semantics. `PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES` does not use one universal placeholder. Category-like columns are zero-filled. Sentinel-style columns that need a true “missing” state use `-1`. That distinction sounds minor until you try to use the same tensor both for embedding lookup and for masking. A structure ID of zero can safely mean “other/no structure bucket”; an AST depth of `-1` can safely mean “not present.” Mixing those two ideas would force each consumer to re-derive intent from context.

The same principle applies to relation edges. `call_edges` and `type_edges` are structurally sparse, but their downstream interpretation has to stay narrow. The POC normalized edge pairs early and kept the chunk-level graph representation separate from token-aligned channels so that structure-aware modules could choose the representation they actually needed. A relation-bias path may care about typed adjacency; a structure embedding path may only care about token-local channels. Compatibility got easier once those consumers were no longer forced through one overloaded metadata blob.

The structure consumers show why this matters. `structure_embeddings.py` expects stable per-token channels for structure IDs, dependency levels, AST depth, sibling index, and AST node type. `relation_bias.py` expects typed relation signals that can become attention bias. Neither module should have to know whether the current batch came from a fully enriched row, a partially enriched row, or an older schema wave. By the time tensors reach them, the contract has to be canonical.

In practice, the POC schema evolved around five recurring metadata families:

| Family | Examples | Why it changed over time |
|---|---|---|
| Token core | `token_ids` | stable from the start |
| Structure channels | structure IDs, AST depth, sibling index | added as structure-aware training matured |
| Chunk segmentation | chunk starts, ends, kinds, dep levels | added to support chunk-level pooling and Tree-style paths |
| Relation edges | call edges, type edges | added once graph signals became explicit |
| Provenance/platform | symbol IDs, type refs, platform IDs | additive metadata used by specialized consumers |

The key is that those families were allowed to appear incrementally without redefining the meaning of existing ones.

```python
PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES = {
    "token_structure_ids": 0,
    "token_dep_levels": 0,
    "token_ast_depth": -1,
    "token_sibling_index": -1,
    "token_ast_node_type": -1,
}
```

That pattern is more important than the exact field list. It says missing enriched metadata must degrade into a deterministic tensor contract, not into branchy Python logic.

## How it lands in MegaCpp

MegaCpp takes the same boundary discipline but narrows the public surface further. The POC proved that it is useful to retain a rich working schema while exposing a smaller loader contract to the actual model code.

For production that means three practical rules.

First, schema names stay explicit and close to their source-of-truth definitions. The packed-row schema layer and the tokenized-enriched schema layer act as source-of-truth modules, not as loose conventions spread across loaders and trainers. If a new relation edge type is added, it lands through those definitions first.

Second, loaders stay tolerant but not ambiguous. Compatibility in MegaCpp does not mean “accept anything.” It means older rows can still load because missing enriched tensors have predefined canonical fills, and newer rows can still load because additive fields do not redefine existing semantics. That lets training jobs roll forward without recooking every stored shard the moment a new structure channel is introduced.

Third, model consumers only read the canonical form. `StructureEmbedding`-style modules and relation-bias modules should consume tensors that are already shape-stable and meaning-stable. The production stack should not carry forward the producer-side ambiguity about whether graph edges arrived as dict pairs, tuples, JSON blobs, or absent columns. The POC already normalized those shapes earlier in the pipeline for exactly this reason.

This is also where structure IDs and relation edges become strategically important. Structure IDs are cheap, dense, and easy to keep backward-compatible. Relation edges are higher-value but also easier to drift because they live at chunk granularity and often start as sparse graph data. The correct production pattern is to keep the richer graph source around within the build pipeline while exporting a narrow, canonical edge representation to loaders.

Another lift from the POC is that compatibility decisions should be testable without replaying an entire corpus build. If a new packed-row field cannot be validated by loading older rows and newer rows through the same canonicalizer, then the change is too coupled. The right place to pay complexity is in schema and normalization code, not in manual migration rituals.

## Ablations and what we kept

Several schema ideas looked attractive and did not survive contact with loader reality.

One idea was to let downstream consumers infer optional metadata from whatever columns happened to be present. That saved a few lines of explicit schema code and cost far more in hidden coupling. The model stack became responsible for understanding producer variation. We did not keep that.

Another idea was to collapse all missing metadata to zero. That worked for categorical channels and failed for sentinel-style numeric fields. AST depth, sibling index, and node type need a real “unknown/not present” representation distinct from a valid bucket. We kept typed fallback values instead.

We also learned not to blur token-aligned and chunk-level metadata. Tokenized enriched rows contain both, but their lifecycles differ. Token-aligned structure channels are usually dense and straightforward to batch. Relation edges and chunk boundaries are sparse and structural. Keeping them as separate schema families made compatibility work much easier.

Finally, we kept normalization close to the producer boundary. `_normalize_graph_edge_pairs(...)`, token-span extraction, and chunk remapping all happen before the data is treated as batch-ready. That is the right tradeoff. If the model stack has to keep re-normalizing rows, then the schema is not really stable.

We also kept the idea that compatibility is directional but should feel symmetric to consumers. Newer code should load older rows through fallback fills. Older code should ignore additive fields it does not understand. That sounds obvious, but it only works when schema authors resist the temptation to reuse an old field name for a new meaning. Most painful migrations start exactly there.

The broad ablation result is simple: additive schema evolution plus canonical fallback fills worked; implicit inference and overloaded field meanings did not.

## Production checklist

- Define every new packed-row or tokenized-enriched field in the schema modules first.
- Keep token-aligned columns, chunk metadata, and row-core fields as separate families.
- Additive columns are fine; changing the meaning of an existing column is not.
- Use typed fallback fills. Zero is not a universal missing-value representation.
- Normalize graph edge encodings before rows are treated as batch-ready.
- Do char-to-token mapping from tokenizer offsets, not token-count heuristics.
- Keep model consumers on canonical tensors only; do not let them inspect raw row variation.
- When adding structure IDs or relation edge variants, make sure older rows still materialize to valid dense tensors.
- Prefer compatibility through deterministic filling over compatibility through branching.

## References

- the offline tokenized-enrichment step
- the tokenized-enriched schema layer
- the packed-row schema layer
- `structure_embeddings.py`
- `relation_bias.py`
- `tokenized-enriched-pipeline-on-tpu.md`
- `structure-embeddings-and-relation-bias.md`
