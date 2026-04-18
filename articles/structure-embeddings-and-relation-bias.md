---
title: "Structure Embeddings and Relation Bias: Teaching the Model That Code Has Shape"
description: "How per-token structure IDs, chunk boundaries, and call/type edges become input embeddings and attention bias in the MegaCpp stack — what we proved in the prototype, what we kept, and what ships in production."
date: "2026-04-18"
tags: ["structure-aware", "c++", "embeddings", "attention-bias", "training"]
---

C++ source is not a flat token stream. It has a preamble, functions, classes, call edges, type edges, and a dependency order imposed by headers. For two training years we watched the model rediscover that structure poorly from whitespace and identifiers alone. This post is about the two features we built to put the structure in the input instead of hoping the model infers it: learned **structure embeddings** added at the input, and a **relation bias** added to attention logits. It also tells the honest story of what survived ablations and what we dropped on the way into production.

## Why the MegaCpp production codebase cares about this

A C++ corpus that is enriched at pre-tokenization time carries a lot of cheap supervision: structure categories per character (preamble, function body, class member, comment, typedef, namespace), AST depth and sibling index from tree-sitter, the node type of the token's surrounding AST node, chunk boundaries with dependency levels computed from includes and type-uses, and cross-chunk edges (caller/callee, type dependency). All of that is computable once in a Rust chunker plus a tree-sitter pass and stored in the enriched parquet schema we ship to the dataloader.

The question is what to do with it at train time. Two shapes of feature dominate: **per-token scalars** (structure category, dep level, AST depth, sibling index, node type bucket) and **per-chunk-pair relations** (call, type, same-level, adjacent-level, preamble-to-code). The first naturally becomes additive input embeddings. The second naturally becomes an ALiBi-style additive attention bias. We built both in our research stack and kept the one that paid rent.

## What we built in the POC

The input-side piece is a structure-embedding layer. Its category set covers the nine character-level structure labels emitted by our Rust chunking pipeline: preamble, function signature, function body, class declaration, class member, comment, typedef, namespace, and other. Its relation taxonomy is deliberately richer than a binary edge marker: caller-to-callee and callee-to-caller are separate planes, as are type-uses versus type-used-by, and the two dependency-level relations (same depth, adjacent depth) let the model see the structural spine of the translation unit.

The main input-level module takes up to five per-token ID streams — structure, dependency level, AST depth, sibling index, and AST node type — and produces a `(B, T, n_embd)` tensor that gets added to the standard token and position embeddings. The implementation went through several rewrites. The current version uses one concatenated embedding vocabulary rather than five separate lookups, a learned low-rank bottleneck, a linear up-projection to model dimension, and per-component learned scalar scales. All weights are zero-initialized, so attaching the module to an existing checkpoint is a step-zero no-op and the signal has to be earned by gradient descent. The configuration can enable every component, only the core always-available features, or an explicit subset; offsets and clamp bounds are precomputed so the forward path stays shape-static and XLA-friendly.

The per-token AST features come from a tree-sitter-cpp pass. We parse each C++ source file, walk the AST to paint per-character arrays for depth, sibling index, and a node-type bucket, then downsample to token level by sampling at each token's first character. The node-type bucketing mirrors the Rust chunking side: ten coarse ranges for declarations, statements, expressions, types, literals, operators, parameters, scope qualifiers, and miscellany. Keeping the Python and Rust mappings bit-identical is a recurring maintenance tax, but it is the only way the same training graph sees the same integers regardless of which enrichment path produced the dataset.

The enriched training schema carries the per-token structure IDs, dependency levels, AST depth, sibling index, and AST node-type buckets, plus chunk boundaries, chunk dependency levels, and edge lists for calls and type dependencies. The loader helpers are small but load-bearing: they coerce mixed array encodings into aligned sequences and keep training ingest from doing that work ad hoc.

The attention-side piece is a relation-bias table with shape `(num_relation_types, num_heads)` — nine relations times the head count, so only a few hundred parameters. The forward path takes a chunk-level relation mask `(B, R, C, C)` built from the edge lists, plus a `(B, T)` token-to-chunk mapping. It combines relation planes into a per-head chunk-level bias with an einsum, then promotes that to `(B, H, T, T)` for attention. Invalid tokens, meaning those outside any chunk, are masked out. The table starts at zero so existing checkpoints reload cleanly, and the bias is added to attention logits before softmax, just like ALiBi. The chunk-level intermediate matters: at roughly `C = 64` and `T = 4k`, the memory cost is tiny compared with a full token-token relation tensor.

There is more in `structure_embeddings.py` than we shipped. A `PlatformEmbedding` module uses `EmbeddingBag` to inject a multi-hot per-document platform vector (OS, GPU family, toolchain) added across all tokens. There is also a `StructureGraphEnricher` — a small TreeFFN that pools tokens to chunks, runs a learned message-passing GNN over the call/type/dep edges, and scatters back to tokens. Both are in the research code, behind flags.

## How it lands in production

The production port is deliberately narrower than the POC surface. In the MegaCpp production-codebase `embedding.py` for the structure feature, `CppMegaStructureEmbedding` is a minimal re-implementation of the input-level piece: the same stacked-embedding plus low-rank-bottleneck design, the same zero-init, the same `"all"` / `"core"` / CSV component spec. It lives next to Megatron's token embedding and is added into the residual before the first block. The config surface in the matching `config.py` is fail-closed: `StructureConfig.from_args` validates `bottleneck_dim`, AST depth caps, sibling index caps, node type counts, and dropout; `tree_ffn_steps` must be positive; `tree_ffn_dropout` must be in `[0, 1)`.

`structure_batch.py` in the MegaCpp Megatron layer is the ingest glue. It lifts the five structure id tensors out of the Megatron batch and threads them into the custom embedding via `set_cppmega_structure_inputs`, keyed by the same column names the POC dataloader uses. One schema, two stacks.

Three things from the POC did not cross the boundary:

- `RelationBiasComputer` and the `chunk_bias_info` plumbing. We ripped out the entire relation-bias dead code in the POC after its last ablation didn't pull weight and MoD-wrapped layers were crashing on it (the wrapped attention sees a compacted `(B, T')` shape; the bias still wanted `(B, T)`). The class is gone from `structure_embeddings.py`, `relation_bias_enabled` is gone from the model config, and the dedicated test file was deleted. `RelationBias` the module still exists as a clean piece of code, but it is not wired into the default stack in the POC and is not ported into MegaCpp. The config flag exists in `StructureConfig` but ships as `False`.
- `StructureGraphEnricher` / TreeFFN. It works, it was measured on both A100 and TPU v6e at small ablation geometries, but once you turn on enriched data plus structure embeddings plus ngram hash, the delta from TreeFFN was smaller than the cost of carrying an extra message-passing graph through every step. Flag lives, default is off.
- `PlatformEmbedding`. Dropped from the MegaCpp seam entirely. The production corpus is narrower than the POC sweep and the platform multi-hot added parameters without measurable help.

What does get lifted as-is is the **input-level structure embedding** with the stacked bottleneck, gated by a feature flag and defaulting to the `"core"` component set (structure + dep_level). What gets rewritten is the Megatron integration layer — the POC used our own forward entry points, whereas MegaCpp injects the additive embedding alongside Megatron's token embedding and makes the field fail-closed in config. What becomes a kernel path is nothing, deliberately: this is cheap enough that we have no reason to push it into Triton or Pallas.

## Ablations and what we kept

The two structure-aware features split cleanly on whether they survived ablation:

| Feature | Module | Ports to MegaCpp | Default in prod | Why |
|---------|--------|------------------|-----------------|-----|
| Input-level structure embedding (core: structure + dep_level) | `structure_embeddings.py` | yes | on | Largest single win in the enriched-data table |
| Stacked single-lookup bottleneck (dim=64) | `structure_embeddings.py` | yes | on | Cuts param count and ~12 kernel launches/step |
| TreeFFN graph enricher | `structure_embeddings.py` (`StructureGraphEnricher`) | flag only | off | Marginal once enriched data + ngram hash are on |
| Relation bias (chunk pair -> per-head logit add) | `relation_bias.py` | unwired | off | Marginal in ablations, breaks under MoD compaction |
| `PlatformEmbedding` multi-hot | `structure_embeddings.py` | dropped | n/a | Production corpus narrower than POC sweep |

We ran the structure-aware features across three overlapping experiments: a no-enrichment baseline, a structure-core rung, and a full stack (structure + tree_ffn + relation_bias + ngram hash). The enriched data is consistently the largest single win in the training-throughput and loss tables, and most of that win comes from the input-level embeddings, not the graph or the attention bias. Two concrete observations from the changelog that shaped the port:

- The single-lookup stacked embedding with a 64-dim bottleneck cut structure-embedding parameter count several-fold and removed roughly a dozen kernel launches per step. Earlier versions used separate embeddings per component, a softmax over component weights, and a mask for absent components. The softmax path also had a silent fp32 upcast via `torch.zeros` in the wrong dtype, which was costing us bf16 throughput. Both are gone.
- `StructureGraphEnricher.forward` went through two quadratic rewrites. The first version materialised a per-token relation mask; the second used `searchsorted` plus `scatter_add` for sparse chunk-to-chunk message passing; the third decomposed the pairwise projections to avoid a trace-variant explosion under Dynamo. The final version works, but it was a lot of engineering for a feature we ended up flagging off in production.

On the attention-bias side, the story is shorter. In the small-model ablation runs, relation bias was measurable but marginal compared to the rest of the enriched stack, and it broke cleanly under MoD compaction. The cost/benefit wasn't there. We kept the module as documented, unwired code so anyone who wants to resurrect the experiment on a future preset can do so without redoing the einsum/gather work.

## Production checklist

The minimum config surface in MegaCpp:

```python
from embedding import StructureConfig

cfg = StructureConfig.from_args(
    enabled=True,
    active_components="core",   # structure + dep_level
    bottleneck_dim=64,
    relation_bias_enabled=False,  # unwired in prod
    tree_ffn_enabled=False,
)
```

- `StructureConfig` must be built via `from_args`; raw construction bypasses validation.
- The dataloader must emit all five token id columns (`token_structure_ids`, `token_dep_levels`, `token_ast_depth`, `token_sibling_index`, `token_ast_node_type`) even when only the core components are active. Missing columns should fail loud, not fall back silently.
- Node-type bucketing must stay bit-identical between Python (`tree_sitter_features.py`) and the Rust chunker. Any change is a schema change and needs a corpus rebuild.
- Keep the structure embedding zero-initialised in checkpoint conversion tools. Accidentally non-zero values at step zero shift the loss curve and make ablation results non-comparable.
- `relation_bias_enabled` and `tree_ffn_enabled` must default to false in MegaCpp configs. Anyone flipping them on is running an ablation, not production.
- `structure_ids` is 9 classes today. If you add a category, bump the Rust enum, the Python enum, and the training parquet version at the same time.
- The chunk-level bias path, if ever re-enabled, is incompatible with MoD's compacted-token path. Disable MoD on any layer that also enables relation bias.

## References

- `structure_embeddings.py` — input-level structure + dep-level embeddings, stacked bottleneck, platform embedding, TreeFFN
- `relation_bias.py` — per-relation-type, per-head additive attention bias
- `tree_sitter_features.py` — per-token AST depth, sibling index, node-type bucket extractor
- the tokenized-enriched schema layer — enriched parquet column names and loaders
- `embedding.py` (MegaCpp structure feature) — production port of the input-level embedding
- `config.py` (MegaCpp structure feature) — fail-closed config surface
- `structure_batch.py` (MegaCpp megatron layer) — batch ingest glue
- [Attention with Linear Biases Enables Input Length Extrapolation — Press et al., ICLR 2022]
- [tree-sitter — Brunsfeld et al.]
