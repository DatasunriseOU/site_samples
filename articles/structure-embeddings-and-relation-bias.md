---
title: "Structure Embeddings and Relation Bias: Teaching the Model That Code Has Shape"
description: "How per-token structure IDs, chunk boundaries, and call/type edges become input embeddings and attention bias in the MegaCpp stack, what the ablations kept, and what ships in deployment."
date: "2026-04-18"
tags: ["structure-aware", "C++", "embeddings", "attention-bias", "training"]
---

C++ source is not a flat token stream. It has a preamble, functions, classes, call edges, type edges, and a dependency order imposed by headers. For two training years we watched the model rediscover that structure poorly from whitespace and identifiers alone. This post is about the two features we built to put the structure in the input instead of hoping the model infers it: learned **structure embeddings** added at the input, and a **relation bias** added to attention logits. It also tells the honest story of what survived ablations and what we dropped on the way into deployment.

## Why the deployed MegaCpp stack cares about this

A C++ corpus that is enriched at pre-tokenization time carries a lot of cheap supervision: structure categories per character (preamble, function body, class member, comment, typedef, namespace), AST depth and sibling index from tree-sitter, the node type of the token's surrounding AST node, chunk boundaries with dependency levels computed from includes and type-uses, and cross-chunk edges (caller/callee, type dependency). All of that is computable once in a Rust chunker plus a tree-sitter pass and stored in the enriched parquet schema we ship to the dataloader.

The question is what to do with it at train time. Two shapes of feature dominate: **per-token scalars** (structure category, dep level, AST depth, sibling index, node type bucket) and **per-chunk-pair relations** (call, type, same-level, adjacent-level, preamble-to-code). The first naturally becomes additive input embeddings. The second naturally becomes an ALiBi-style additive attention bias. We prototyped both and kept the one that paid rent.

## What we built in the training prototypes

The input-side piece is a structure-embedding layer. Its category set covers the nine character-level structure labels emitted by our Rust chunking pipeline: preamble, function signature, function body, class declaration, class member, comment, typedef, namespace, and other. Its relation taxonomy is deliberately richer than a binary edge marker: caller-to-callee and callee-to-caller are separate planes, as are type-uses versus type-used-by, and the two dependency-level relations (same depth, adjacent depth) let the model see the structural spine of the translation unit.

The main input-level module takes up to five per-token ID streams — structure, dependency level, AST depth, sibling index, and AST node type — and produces a `(B, T, n_embd)` tensor that gets added to the standard token and position embeddings. The implementation went through several rewrites. The current version uses one concatenated embedding vocabulary rather than five separate lookups, a learned low-rank bottleneck, a linear up-projection to model dimension, and per-component learned scalar scales. All weights are zero-initialized, so attaching the module to an existing checkpoint is a step-zero no-op and the signal has to be earned by gradient descent. The configuration can enable every component, only the core always-available features, or an explicit subset; offsets and clamp bounds are precomputed so the forward path stays shape-static and XLA-friendly.

The per-token AST features come from a tree-sitter-cpp pass. We parse each C++ source file, walk the AST to paint per-character arrays for depth, sibling index, and a node-type bucket, then downsample to token level by sampling at each token's first character. The node-type bucketing mirrors the Rust chunking side: ten coarse ranges for declarations, statements, expressions, types, literals, operators, parameters, scope qualifiers, and miscellany. Keeping the Python and Rust mappings bit-identical is a recurring maintenance tax, but it is the only way the same training graph sees the same integers regardless of which enrichment path produced the dataset.

The enriched training schema carries the per-token structure IDs, dependency levels, AST depth, sibling index, and AST node-type buckets, plus chunk boundaries, chunk dependency levels, and edge lists for calls and type dependencies. The loader helpers are small but load-bearing: they coerce mixed array encodings into aligned sequences and keep training ingest from doing that work ad hoc.

The attention-side piece is a relation-bias table with shape `(num_relation_types, num_heads)` — nine relations times the head count, so only a few hundred parameters. The forward path takes a chunk-level relation mask `(B, R, C, C)` built from the edge lists, plus a `(B, T)` token-to-chunk mapping. It combines relation planes into a per-head chunk-level bias with an einsum, then promotes that to `(B, H, T, T)` for attention. Invalid tokens, meaning those outside any chunk, are masked out. The table starts at zero so existing checkpoints reload cleanly, and the bias is added to attention logits before softmax, just like ALiBi. The chunk-level intermediate matters: at roughly `C = 64` and `T = 4k`, the memory cost is tiny compared with a full token-token relation tensor.

There is more in `structure_embeddings.py` than we shipped. A `PlatformEmbedding` module uses `EmbeddingBag` to inject a multi-hot per-document platform vector (OS, GPU family, toolchain) added across all tokens. There is also a `StructureGraphEnricher` — a small TreeFFN that pools tokens to chunks, runs a learned message-passing GNN over the call/type/dep edges, and scatters back to tokens. Both are in the research code, behind flags.

## How it lands in production

The deployed port is deliberately narrower than the prototype surface. In the MegaCpp structure feature, `CppMegaStructureEmbedding` is a minimal re-implementation of the input-level piece: the same stacked-embedding plus low-rank-bottleneck design, the same zero-init, the same `"all"` / `"core"` / CSV component spec. It lives next to Megatron's token embedding and is added into the residual before the first block. The configuration surface validates bottleneck size, AST depth caps, sibling index caps, node type counts, and dropout so the feature stays well-bounded.

`structure_batch.py` in the MegaCpp Megatron layer is the ingest glue. It lifts the five structure id tensors out of the Megatron batch and threads them into the custom embedding through the same column names the prototype dataloader uses. One schema, two stacks.

Three things from the earlier implementation did not cross the boundary:

- `RelationBiasComputer` and the `chunk_bias_info` plumbing. We removed the full relation-bias path after its last ablation stopped paying for itself and token-compacted attention paths exposed a shape mismatch. `RelationBias` remains a standalone module, but it is not part of the default MegaCpp path.
- `StructureGraphEnricher` / TreeFFN. It worked in ablations, but once enriched data plus structure embeddings plus ngram hash were enabled, the extra gain was smaller than the cost of carrying message passing through every step.
- `PlatformEmbedding`. It did not earn its place on the production corpus and is not part of MegaCpp.

What does carry forward is the **input-level structure embedding** with the stacked bottleneck, gated by a feature flag and defaulting to the `"core"` component set (structure + dep_level). MegaCpp injects that additive embedding alongside the token embedding path. It stays a regular model feature rather than becoming a custom kernel path because the cost is already low.

## Ablations and what we kept

The two structure-aware features split cleanly on whether they survived ablation:

| Feature | Module | Ports to MegaCpp | Default in prod | Why |
|---------|--------|------------------|-----------------|-----|
| Input-level structure embedding (core: structure + dep_level) | `structure_embeddings.py` | yes | on | Largest single win in the enriched-data table |
| Stacked single-lookup bottleneck (dim=64) | `structure_embeddings.py` | yes | on | Cuts param count and ~12 kernel launches/step |
| TreeFFN graph enricher | `structure_embeddings.py` (`StructureGraphEnricher`) | experimental | off | Marginal once enriched data + ngram hash are on |
| Relation bias (chunk pair -> per-head logit add) | `relation_bias.py` | no | off | Marginal in ablations, breaks under token compaction |
| `PlatformEmbedding` multi-hot | `structure_embeddings.py` | no | n/a | Production corpus did not justify the extra parameters |

We ran the structure-aware features across three overlapping experiments: a no-enrichment baseline, a structure-core rung, and a full stack (structure + tree_ffn + relation_bias + ngram hash). The enriched data is consistently the largest single win in the training-throughput and loss tables, and most of that win comes from the input-level embeddings, not the graph or the attention bias. Two concrete observations from the ablation record shaped the port:

- The single-lookup stacked embedding with a 64-dim bottleneck cut structure-embedding parameter count several-fold and removed roughly a dozen kernel launches per step. Earlier versions used separate embeddings per component, a softmax over component weights, and a mask for absent components. The softmax path also had a silent fp32 upcast via `torch.zeros` in the wrong dtype, which was costing us bf16 throughput. Both are gone.
- `StructureGraphEnricher.forward` went through several rewrites to cut down quadratic work. The final version works, but it remained too expensive for the value it added in production settings.

On the attention-bias side, the story is shorter. In the small-model ablation runs, relation bias was measurable but marginal compared to the rest of the enriched stack, and it broke under token compaction. The cost-benefit was not there, so it stays out of the default MegaCpp path.

## Deployment checklist

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

- `StructureConfig` should be built through validated configuration entry points rather than ad hoc construction.
- The dataloader should emit all five token id columns (`token_structure_ids`, `token_dep_levels`, `token_ast_depth`, `token_sibling_index`, `token_ast_node_type`) even when only the core components are active.
- Node-type bucketing must stay consistent across preprocessing and training.
- Keep the structure embedding zero-initialised in checkpoint conversion tools. Accidentally non-zero values at step zero shift the loss curve and make ablation results non-comparable.
- `relation_bias_enabled` and `tree_ffn_enabled` should default to false in MegaCpp configs.
- The chunk-level bias path, if ever re-enabled, should not be combined with token compaction in the same layer without re-validation.

## References

- [Compile commands and semantic graphs](https://megacpp.com/blog/compile-commands-and-semantic-graphs.md)
- [Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
