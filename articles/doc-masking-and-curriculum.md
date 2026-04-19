---
title: "Document Masking and the Curriculum: What to Feed Each Specialist First"
description: "Why MegaCpp masks documents inside packed sequences, how the four-phase curriculum is shaped from 4K syntax to 64K repository graphs, and what our ablations told us about the right starting diet for each specialist."
date: "2026-04-18"
tags: ["curriculum", "doc-masking", "long-context", "training", "c++"]
---

If you train a code model long enough on packed sequences without document masking, it will eventually learn the wrong lesson: that the function in this file might secretly know about the class three documents back in the same packed row. At 4K context that is a noise problem; at 64K it is a correctness problem. This post explains why we mask documents end-to-end at MegaCpp, how the resulting four-phase curriculum is shaped, and what the ablations told us about the order in which different specialists should be fed.

## Why mask documents at all

Sequence packing is non-negotiable on hardware that bills you per second. We pack multiple short documents — functions, classes, commits, file slices — into a single fixed-length training sequence to push Model Flops Utilization toward 100% instead of paying explicit pad tokens. The catch is that vanilla causal self-attention, given a packed row of `[doc A | doc B | doc C]`, lets every token in B attend to every token in A. There is no architectural reason it should not — the tokens are right there in the same row.

On 4K pretraining the cost of this contamination is small. There are usually only a handful of documents per row, and the model can mostly average over the noise. On 16K and 64K it is fatal. A 64K row commonly carries a dozen unrelated documents from completely unrelated repositories, and unmasked attention will teach the model that any function might depend on any other function in the same packed window. That is the opposite of repository-level reasoning. We measured this directly and refuse to ship a long-context training run without document masking.

The mechanism is straightforward in concept. Every document gets a leading BOS token (the tokenizer is configured to enforce this), and a `doc_ids` tensor is computed at the start of `GPT.forward()` as a cumulative sum over BOS positions in `input_ids`. Two tokens with the same `doc_id` may attend to each other causally; two tokens with different `doc_ids` may not. The implementation cost in storage is zero because we infer `doc_ids` from `input_ids` instead of materializing them in the parquet schema.

## End-to-end means every layer, not just attention

The naive read of "document masking" is "mask attention." That is necessary but not sufficient. Our model uses several layer families, and each one can leak across documents in its own way:

1. Attention. This is the obvious one. We dispatch the mask per backend: FlexAttention composes a `document_causal_mask` with the existing softcap `score_mod` and runs block-sparse, so the MFU cost on 64K is small. FA3 varlen converts `doc_ids` into `cu_seqlens` and runs unpadded; we unpad to flat `(total_tokens, H, D)`, call `flash_attn_varlen_func`, and re-pad. SDPA materializes a 2D mask but is gated to T ≤ 8192 because the `O(T²)` mask is unusable above that (256 MB per sample at 16K, 4 GB at 64K). On TPU, both Pallas FlashAttention and JAX Splash Attention accept `segment_ids` directly at the kernel level — no `O(T²)` mask materialization, and Splash additionally fuses softcapping.
2. Mamba SSM state. The hidden state from document A bleeds into document B unless we zero it at boundaries. On CUDA we zero state mid-sequence; on TPU we integrate the reset into the XLA scan body as a multiplicative carry mask, because the scan is a compiled loop and cannot simply be stopped and restarted at a boundary.
3. Mamba conv1d. The `conv1d` kernel size is 4. Each output depends on the three preceding tokens, which at a boundary belong to a different document. We mask the conv1d input buffer at boundaries so the kernel does not pull in cross-document context.
4. Engram and the Manifold Hyperbolic Convolution mixer. Both can in principle accumulate cross-document state. We treat them as a Phase-2 concern and verify with the same boundary tests.

The success criterion is not "the loss looks better." It is mathematical: tests assert that attention scores between different `doc_ids` are exactly `-inf` (or `0.0` post-softmax), and that Mamba SSM state and conv1d output are bitwise zero across boundaries. There is also an attention-leakage integration test: pack `[A, A, A]` and `[B, B, B]` together, run the forward, and assert that the output for the B tokens is mathematically identical to a run where B was processed alone. If that test fails, no other progress matters.

The MFU budget we hold is "no more than 3-5% drop versus naive packing." FlexAttention's `BlockMask` block-sparsity and the TPU Pallas backends' `segment_ids` make this achievable in practice, and we monitor it on every long-context run.

## The four-phase curriculum

With masking in place, the curriculum is what teaches each specialist to reason. We use four phases of progressively longer context, mapped to dataset families produced by different stages of the data pipeline.

Phase 1 — Syntax Mastery (4K context). The goal is C++ syntax, basic structures, and short-range dependencies. The datasets are simple and doc-rich short-context mixes — pre-tokenized binary shards containing the modified files *without* deep call graphs, dense in code edits and structured comments. The later structured variants use a generated header (Doxygen-style or simple), a `// === CHANGES ===` block showing removed lines as comments and added lines as live code, and a `// === FULL FILE (Post-Commit) ===` block with the resulting state. Sequences are packed to 4096 tokens via a memmap over `uint16` (or `uint32` for the 131K vocab) binaries.

Phase 2 — File-Level Reasoning (16K context). The goal is to learn how functions and classes inside the same file relate. The dataset is a file-context graph mix — Tree-sitter-built call graphs structured as `Target → Direct Callers → Direct Callees`, packed to 64K bounds but cropped to 16,384 tokens at dataloading. Because the packing algorithm keeps the most critical context closest to the modification, the 16K window captures file-local and immediate cross-file dependencies cleanly.

Phase 3 — Repository Graph Reasoning (64K context). The goal is full project-level awareness — variables, inheritance, side effects across multiple files. The dataset is the full file-context graph mix plus a build-resolved semantic graph mix, where the latter is built on top of the repository compile database for high-accuracy semantic relationships. Up to 64,000 tokens of heavily interconnected C++ code are packed, tracing deep semantic links (callers of callers, base interfaces) injected immediately before the target modification.

Phase 4 — Structure-Aware Training (all context lengths). The goal is to teach the model code structure (call graphs, type hierarchies, scope boundaries) through learnable embeddings and attention bias rather than through flat text alone. The dataset is an enriched structure-aware mix — the same compilable C++ as Phases 2-3, but in enriched parquet with `structure_ids`, `chunk_boundaries`, `call_edges`, `type_edges`, `ast_depth`, `sibling_index`, and `ast_node_type` columns. The model receives structure and dep-level embeddings at the input layer and learned per-relation attention bias inside attention. Phase 4 can run concurrently with earlier phases because enriched data is backward compatible — drop the columns and you have flat text.

The order matters. Trying to teach repository graph reasoning to a model that has not yet learned syntax produces a model that is confidently wrong about both. Trying to teach syntax to a model that has already been overfit on long graphs wastes the long-graph data. Curriculum is not a nice-to-have; it is the cheapest way we know to use a fixed compute budget well.

## What the ablations told us

We ran a battery of ablations to size the per-phase data mix per specialist. Three findings were robust enough to act on.

First, Phase 1 dominates early loss for every specialist, but the *kind* of Phase 1 data matters per specialist. The Algorithms specialist responds best to the simpler 4K shards — tight self-contained functions with clear pre/post conditions. The Templates specialist needs the structured diff variant more, because template-heavy commits often touch multiple deeply-nested constructs and the structured diff makes the change site explicit. The Build/Toolchain specialist responds to neither well at 4K — most build-system pain only appears at file scope, so it benefits from skipping ahead to Phase 2 sooner.

Second, the long-context win is not uniform. Specialists whose work is genuinely repository-level — Orchestration (multi-file refactors), Build (cross-target dependency tracking), Service-Framework — show large `val_bpb` improvements going from Phase 2 to Phase 3, on the order we expected. Specialists whose work is mostly local — Algorithms, Memory/RAII, Templates — show much smaller improvements at Phase 3 and would be wasted on a long-only diet. We use this to *bias* the per-specialist mix at Phase 3, not to skip it; even local-work specialists benefit from seeing some 64K context because production code often forces local logic to depend on a far-away header.

Third, structure-aware Phase 4 helps most where the model already understands the syntax. Adding structure embeddings on top of an under-trained Phase 1 model is a wash — the model has no foundation to anchor the structural signal to. The same Phase 4 data, applied after the model has converged on Phase 2, produces a clear improvement on tasks where the answer depends on knowing whether a token is inside a `class_decl` versus a `func_body` versus a `typedef`. The Templates specialist gets the largest absolute gain from Phase 4, because the legality of a `requires`-clause depends entirely on its structural position, and the `structure_ids` column is exactly the signal needed.

## Why we mask documentation specifically

There is a separate decision about what to do with documentation comments — Doxygen blocks, javadoc-style headers, large block comments. Comments are 24.2% of the corpus by bytes. Stripping them entirely is wrong: we measured that it makes the model noticeably worse at writing readable code, because it never sees the natural-language explanations engineers actually attach to functions.

But in some contexts we do want the model to learn from documentation without being able to *cheat* off it. The clearest case is fill-in-the-middle training on a function body, where the function has a Doxygen header that describes exactly what the body should do. Without intervention the model learns to copy from the docstring instead of synthesizing the body from first principles. The fix is to mask the loss on the body when the header is present, or to mask the header when the body is the target — both conditioned on whether we want the model to learn the documentation→code direction or the code→documentation direction. In practice we run both directions, in a roughly even split, and we never compute loss on a target that the input has trivially leaked.

The same masking machinery that handles document boundaries handles this — it is just a finer-grained `loss_mask` rather than a finer-grained `doc_ids`. The packed-row parquet schema reserves columns for this (`input_ids`, `target_ids`, `loss_mask` are all required at runtime length T; `doc_ids`, `valid_token_count`, `pack_id`, `num_docs` are recommended) so the contract is uniform across base training, validation, and SFT.

## What we feed each specialist first

Pulling the threads together, the per-specialist starting diet is roughly:

- Algorithms: heavy Phase 1 simple short-context shards, then Phase 2 file-context graph mix to learn when to reach for which container.
- Templates: heavy Phase 1 structured-doc shards, then Phase 4 enriched parquet as soon as it is available.
- Memory/RAII: heavy Phase 1 mixed with Phase 2; Phase 3 useful but not critical.
- Build/Toolchain: light Phase 1, heavy Phase 2 and Phase 3.
- Service-Framework / Orchestration: light Phase 1, heavy Phase 3 and Phase 4 - these live in 64K windows.
- Testing: heavy Phase 1 simple short-context shards with over-sampled test-fixture patterns; Phase 4 helps on distinctive structural shapes.
- Systems/C: balanced across Phase 1-3; macros and RCU patterns benefit from both local and repo-scale awareness.
- Compilers/Toolchain: heavy Phase 2 and Phase 3 over LLVM; pass infrastructure makes no sense without file-level context.

These are starting diets, not fixed recipes. Each specialist is fine-tuned on a domain-skewed mix of the same base corpus afterwards.

## The unsexy summary

Document masking and curriculum are the boring parts of training a code model, and they are also the parts that decide whether 64K context is real or theatre. Masking is end-to-end or it is not real — attention, Mamba SSM state, conv1d, anything with state. Curriculum is empirical or it is cargo-cult — ablate, measure `val_bpb` per phase per specialist, and let the specialists that benefit from long context get it while sparing the ones that do not. The wins are not glamorous, but they compound: a model that has actually learned syntax before it sees a 64K repository graph generalizes; a model that was thrown into the deep end does not.

## Phase-to-data map

| Phase | Context | Datasets | Goal |
|---|---|---|---|
| 1 syntax | 4K | simple and doc-rich short-context mixes | C++ syntax + short-range deps |
| 2 file | 16K | file-context graph mix | file-level reasoning |
| 3 repo | 64K | build-resolved repository mix | repository-level reasoning |
| 4 structure | 64K | enriched structure-aware mix | structure-aware bias |

```python
# doc_ids inferred at GPT.forward() entry - no storage-format change.
is_bos = (input_ids == BOS_ID)
doc_ids = torch.cumsum(is_bos.to(torch.int32), dim=-1)
# attention backend converts to its native form (cu_seqlens / segment_ids / mask)
```

## References

- [MegaCpp public repository](https://github.com/DatasunriseOU/cppmega)
- [MegaCpp public sample pack](https://github.com/DatasunriseOU/site_samples)
- The public masking, curriculum, and data-pipeline notes linked from the MegaCpp repositories.
