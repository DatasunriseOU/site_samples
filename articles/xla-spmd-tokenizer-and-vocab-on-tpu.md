---
title: "Vocab and Tokenizer Plumbing on TPU: What XLA SPMD Makes You Decide Up Front"
description: "Vocab-size constraints under XLA, the padding choices that keep the compile cache stable, sharded embedding init under SPMD, and the per-specialist platform vocab story."
date: "2026-04-18"
tags: ["tpu", "v6e", "xla", "spmd", "tokenizer", "vocab", "embeddings"]
---

The C++ specialist family in this codebase runs the same tokenizer and the same model definition on H200 and TPU v6e. The interesting part is not the tokenizer itself; it is a HuggingFace BPE artefact and a thin Python adapter. The interesting part is everything around it on the TPU path: how the vocab size becomes a static compile constant, how the embedding row count gets padded so the compile cache does not blow up, how the embedding parameter is sharded under SPMD without falling into XLA's propagation traps, and how the per-document platform vocab gets pulled through the dataloader without forcing a synchronization point.

## Why the TPU path has more rules

On the GPU path the tokenizer story is boring: build the BPE, pad the vocab to a multiple of 64 for tensor cores, register the embedding, train. On the TPU path the same sequence runs into three constraints the GPU path mostly ignores. First, the vocab size is part of the XLA HLO graph; a change in `vocab_size` (or in the padded vocab size) is a recompile. Padding decisions that are throwaway on H200 become load-bearing on v6e. Second, the embedding parameter is the largest single tensor in the model on small specialists, sitting in the propagation graph immediately under both `wte` and `lm_head`; whatever sharding XLA infers for one propagates into the other through tied or untied weights, and through the loss path. Third, the tokenizer is not the only vocabulary the dataloader pushes onto the device: a per-document platform vocabulary (113 IDs covering OS, RTOS, GPU, arch, compiler, C++ standard) gets summed into a single embedding per document and added to every token in that document's row.

## The two tokenizers and one padding shim

the tokenizer sample is the generic GPT-style HuggingFace BPE wrapper: a `train_from_iterator(text_iterator, vocab_size)` that builds a BPE with byte-fallback, a GPT-4-style pre-tokenizer regex, a `ByteLevel` decoder, and a fixed special-token list. The pre-tokenizer split was deliberately tuned to `\p{N}{1,2}` rather than `\p{N}{1,3}` because the wider form wastes vocabulary on multi-digit number tokens that small models never recover. `cpp_the tokenizer sample` is the C++-aware variant: a fixed C++ vocabulary (keywords, STL names, multi-char operators) merged with a learned BPE, BERT-style whitespace handling on decode, and the special-token aliases (`<|bos|>` -> `<BOS>`, `<|code_start|>` -> `<CODE_START>`, the `<|think_*|>` family, the tool-calling tokens) that let upstream-style call sites work against the C++ tokenizer transparently.

The vocab size for the target specialist is 65536. Picking a power of two is not aesthetic; it is the largest XLA optimisation we get for free. 65536 is divisible by every TP degree we use (2, 4, 8), pads cleanly to the next tensor-core multiple, and leaves the per-shard vocab dimension at a round size when we shard rows of `lm_head` across the model axis. A vocab of 65000 would have forced a per-rank-uneven shard at TP=8 and a recompile every time the sharding changed.

The padding shim is in the main model runtime module. `GPT.__init__` takes a `pad_vocab_size_to=64` argument and rounds the configured `vocab_size` up to the next multiple before allocating `wte` and `lm_head`:

```python
padded = ((cfg.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
self.wte    = nn.Embedding(padded, cfg.n_embd)
self.lm_head = nn.Linear(cfg.n_embd, padded, bias=False)
# Slice logits back to cfg.vocab_size at the loss boundary.
```

The forward slices logits back to `vocab_size` at the loss boundary so the extra rows never participate in cross-entropy. On the GPU path that pad keeps the matmul on the tensor-core fast path. On the TPU path it does the same job for VMEM tile sizes and, more importantly, makes the embedding shard-friendly: 64 divides every TP degree we use, so a row-sharded `lm_head` always sees an even per-rank slice.

## Embedding sharding under SPMD

There are two annotation surfaces. On the CUDA-native TP path the embedding uses `parallelize_module` with `RowwiseParallel(input_layouts=Replicate(), output_layouts=sp_layout, use_local_output=True)` when sequence-parallel attention is on, the final RMSNorm runs as a SequenceParallel module, and the LM head accepts the seq-sharded input and produces vocab-sharded output via `ColwiseParallel`. Without SP, the embedding stays replicated and the LM head still gets row-sharded across TP ranks. A `_vocab_parallel_enabled` marker attribute is set on `lm_head.weight` so the fused loss path knows it is operating on per-rank vocab shards.

On the XLA SPMD path under `_apply_tensor_parallel_sharding`, the embedding stays replicated across the model axis. The comment in the TPU training launcher is explicit: the LM head row-shard is a CUDA-only optimisation, and the TPU loss path operates on the full vocab tensor with the model-axis collective fused into the cross-entropy. The reason for the asymmetry is that XLA's sharding propagation around the loss is more reliable than the analogous DTensor path on CUDA, and the TPU compile cache is much more sensitive to sharding-spec churn than the eager CUDA path is. We pin both decisions explicitly via `xs.mark_sharding`; an unannotated embedding under propagation is a future precision bug waiting to happen.

## The platform vocab and how it crosses the data-loader boundary

the platform-vocabulary sample defines a flat 113-ID space (ID 0 is padding, IDs 1..112 cover the six categories). The categories share one ID space deliberately so the embedding is a single `EmbeddingBag(113, n_embd, mode="sum", padding_idx=0)`. `MAX_PLATFORM_IDS = 20` is the per-document buffer cap; any document with more than 20 active labels gets truncated, which has not happened on observed organizationus rows but is enforced for buffer-shape determinism.

The dataloader plumbing pulls platform IDs through three call sites in the public dataloader sample. On the parquet path, when the row group has a `platform_ids` column, IDs are read directly; when only the raw `platform_info` JSON is present, `platform_info_to_ids(pi)` walks the six categories and emits a sorted unique list. On the materialisation path, the per-batch buffer `padded_platform = torch.zeros((B, MAX_PLATFORM_IDS), dtype=torch.long, device=device)` is filled row by row with truncation to `MAX_PLATFORM_IDS`. The shape is fixed at construction time so the embedding bag forward pass sees the same `(B, MAX_PLATFORM_IDS)` shape on every step, which is the only way it stays in the XLA compile cache.

The text-prefix variant is the third call site. When the dataloader runs in metadata-prefix mode, `platform_info_to_prefix(...)` builds a single-line C++ comment and the tokenizer encodes it. That path is GPU-eager-friendly but on the TPU path we explicitly forbid it for input rows: `strict_token_only_train` raises if a row arrives without its platform prefix already encoded into `token_ids`. The reason is that runtime tokenisation is variable-length and would force a synchronization point per batch on TPU; the offline materialisation path puts the encoded prefix into the parquet so the TPU dataloader only ever sees fixed-shape token arrays.

## What the TPU dataloader has to guarantee

| Constraint | What it forces |
|---|---|
| Vocab size in the HLO graph | Pad once at model build, never change at runtime |
| Embedding annotation | Pin via `xs.mark_sharding`; never let propagation infer |
| Per-row platform IDs | Pre-padded to `MAX_PLATFORM_IDS`; no synchronization point per batch |
| Token rows | Fixed `(B, T)` int32; no Python-side variable shapes |
| Doc IDs | Materialised at `(B, T)` int32; no dynamic doc boundaries |
| Packing policy | `single_doc_block` by default to keep rows rectangular |
| Shard list | Fixed `index_to_filename` for `shard_NNNNN.parquet` up to the configured max |

The TPU dataloader branch in the public dataloader sample and the public dataset sample is otherwise the same as the CUDA branch: parquet files from a fixed shard list, pyarrow row-group iteration, and a packing policy that the TPU side runs in `single_doc_block` mode by default. The `doc_ids` tensor is materialised at `(B, T)` int32 so the attention masking path on TPU has document boundaries without dynamic shapes.

## What we kept and threw away

We kept the 65536 vocab, the `pad_vocab_size_to=64` shim, the `_vocab_parallel_enabled` marker, the asymmetric "row-shard `lm_head` on CUDA, replicated on TPU" decision, the EmbeddingBag aggregator at 113 IDs with `padding_idx=0`, the `MAX_PLATFORM_IDS=20` cap, and the rule that runtime tokenisation is forbidden in the TPU critical path.

We threw away `\p{N}{1,3}` in the pre-tokenizer regex (wasted vocab), the temptation to let XLA propagate the embedding sharding (precision bug surface), runtime metadata-prefix tokenisation on TPU (synchronization point per batch), variable-length `doc_ids` (dynamic shapes break the cache), and any vocab size not divisible by every TP degree we use.

The throughline is small. On TPU you decide tokenizer plumbing once, statically, and never let a Python-side variable shape into the critical path. Everything else, including the model, mostly takes care of itself.

## How the dataloader keeps the critical path rectangular

The TPU dataloader's job is to produce fixed-shape `(B, T)` token arrays, fixed-shape `(B, T)` `doc_ids`, and a fixed-shape `(B, MAX_PLATFORM_IDS)` platform buffer, on every step, regardless of what the underlying organizationus row looked like. Three pieces of plumbing make that work.

First, the parquet shard list is fixed at construction time. The dataloader iterates row groups in a deterministic order, picks rows in a deterministic packing policy (`single_doc_block` on TPU by default), and pads to `T` with the tokenizer's pad token. The packing policy is what controls the `doc_ids` distribution; `single_doc_block` is the simplest because every row carries one document boundary at position 0 and `doc_ids` is constant across the row. More elaborate packings are GPU-friendly but produce variable doc-id structure that we deliberately keep off the TPU path.

Second, the platform IDs are pre-aggregated. The materialisation path computes `platform_info_to_ids(pi)` once per row at parquet build time and writes the result into the row's `platform_ids` column. The runtime dataloader reads it directly into the per-batch buffer with truncation to `MAX_PLATFORM_IDS`. The runtime never calls `platform_info_to_ids`; that path exists only for the build-time materialisation tools and for unit tests.

Third, the special tokens are pre-encoded. Anything that would otherwise require a runtime tokenizer call (the `<|code_start|>` family, the `<|think_*|>` family, the tool-calling tokens) is encoded at materialisation time and stored as integer IDs in the row. The runtime never invokes the BPE; it reads pre-encoded integer arrays and pads them to the row width.

## What we still cannot do on the TPU path

Two things remain unsupported on the TPU critical path. The first is dynamic vocab expansion. If we add a new domain that needs new BPE merges, the new vocab gets a new `vocab_size` and a recompile; we batch those expansions into scheduled update windows rather than mid-run. The second is per-document tokenizer variants. We have looked at running the C++ tokenizer on C++ rows and the generic tokenizer on prose rows; the runtime branching that would require breaks the rectangular critical path. We instead run a single tokenizer trained on a mixed organizationus and accept the small quality loss on prose.

Both limits are real but small. They are the price for a critical path that compiles once and runs unchanged for the life of the cache.

## What the receipt records about the vocab

Every TPU receipt records the active vocab size, the padded vocab size, the `pad_vocab_size_to` value, the platform vocab size, the `MAX_PLATFORM_IDS` cap, the tokenizer file SHA, and the dataloader packing policy. When a regression appears those fields are usually the first place to look, because a vocab change or a packing change invalidates the compile cache and produces an attribution-ambiguous slowdown that looks at first like a kernel regression.

Recording these fields is cheap. Investigating a regression without them is expensive. We learned the order of operations the hard way and the receipt now reflects it.

## References

- https://megacpp.com/blog/cpp-tokenizer-deep.md
- https://megacpp.com/blog/tokenizer-v2-v3.md
- https://megacpp.com/blog/slm-data.md
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/data/masking_pipeline_excerpt.py
- https://docs.pytorch.org/xla/master/runtime.html
- https://docs.pytorch.org/xla/master/learn/pjrt.html
