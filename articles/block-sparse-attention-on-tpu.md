---
title: "Block-Sparse Attention on TPU v6e: Block Masks, MXU-Friendly Tiles, and the Long-Context Win"
description: "How we make block-sparse and clustered-sparse attention compile cleanly through XLA on TPU v6e: block-mask construction, MXU-aligned tile sizes, the contract tests that catch the silent failures, and the long-context throughput we keep."
date: "2026-04-18"
tags: ["tpu", "xla", "sparse-attention", "pallas", "long-context"]
---

Block-sparse attention is the only way to push a transformer past 64K tokens on a v6e chip without paying for an O(T^2) score matrix. The CUDA path is well-trodden: FlexAttention with a BlockMask, MoBA or a learned indexer for KV blocks, FA3 or CuTe for the inner kernel. The TPU path is its own animal: the inner kernel is a Pallas call, partitioning is GSPMD, and almost every "data-dependent" decision turns into a recompile if you are not careful. This post is what we use for block-sparse and clustered-sparse attention on TPU v6e, what stays on CUDA, and where the contract tests live that prevent us from quietly running dense while convinced we are sparse.

## Why MegaCpp cares about this

Long context is non-negotiable for code-organizationus pretraining. A packed row spans many small documents and a few large ones; the right answer is to attend within and between related documents, not to every byte. Quadratic attention at 64K is ~17 GiB of bf16 scores per chip; at 256K it is comically out of budget on v6e. Sparse attention is the lever that turns long context from research demo into something we can train on. Block-sparse keeps the matmul layout MXU-friendly: the inner kernel still runs dense matmuls over chunks the partitioner can map to systolic tiles.

On the deployed path, the stack ships a fused FP8 sparse DSA path that mirrors the same contracts at very different shapes. TPU and H200 both feed the same evaluation: a long-context win that survives the actual training loop, not just a microbenchmark.

## What we built in the POC

Five modules carry the load. Two define the runtime contract; three implement the inner kernels.

the public block-sparse semantics sample owns the policy that decides which block pairs are valid, full, or partial. `classify_selected_block_masks(selected_block_indices, doc_ids, block_size, seq_len, ...)` returns three masks: `valid_mask` (selected block may contain at least one legal token pair), `full_mask` (safe without inner token mask), `partial_mask` (still needs token-level causal/doc cleanup). The mixed-document policy is conservative: cross-document pairs rejected; mixed-document overlaps allowed only as partial; full-block pairs require strict-past + same-document + homogeneous query/key blocks. `describe_blockized_mixed_document_policy()` restates the rules so the runtime can ship them as a JSON contract.

the public attention-validity sample normalises the validity contract once at the boundary: token-prefix counts, slot-prefix counts, base block tokens. Consumers take an `AttentionValidity` and resolved `SlotFrontier` instead of re-deriving "where does this row's real content end" inside the compiled region.

the public sparse-attention sample (~8.5K lines) owns three semantic branches:

1. **Exact-token DSA**, lightning indexer: each query picks its own key positions. Backends: sdpa (dense-mask), gathered chunked, FA3-gather. TPU uses the gathered fallback; FA3 is CUDA-only.
2. **Blockized sparse CUDA**, MoBA or BlockIndexer: a block-level router selects KV blocks per query block, FlexAttention with a BlockMask handles execution. The CuTe FLASH backend is a CUDA blockized backend, not the dense FA4 path.
3. **Donor / dense FA4** for runtime comparison only.

The exact-token path on TPU calls back into the Pallas FA kernel via the TPU attention dispatch layer's gathered-chunked path. FlexAttention is CUDA-only because its grid construction is Triton-shaped; on TPU we hand block-sparse off to the clustered-sparse path.

the public clustered-sparse attention sample bridges the experimental Pallas pipelines under `experiments/sparse_pallas/` into the model:

- `kernel_config_from_gpt_config` derives a `KernelConfig` with defaults `query_tile_size=256`, `kv_tile_size=1024`, `selected_block_size=256`, `block_stride=64`, `compression_block_size=128`. These compile cleanly on v6e and stay in the MXU sweet spot.
- `_torch_to_jax` / `_jax_to_torch` use dlpack for zero-copy on XLA, sharing PJRT buffers. Same pattern as the Splash bridge.
- `_make_fwd_fn` is cached on `(use_fused_scoring, use_pallas, kernel_config_items, seq_len)`. `call_jax` keys its LRU on Python function identity; without the cache, every forward built a new closure and the partitioner recompiled the whole pipeline (~260 recompiles worst case).
- `_chunked_causal_vjp_fn` chunks the backward in 512-query slices so peak per-iteration memory is `O(C * T)` rather than `O(T^2)`. At T=256K with C=2048 that is ~2.5 GiB instead of 64 GiB.

the public sparse-contract sample is the contract surface. It defines canonical names for semantic branches (`exact_token`, `blockized`, `donor`), execution contracts (`eval_no_grad`, `train`, `runtime_compare`, `whole_model_eval`), and measurement paths. `infer_promoted_sparse_semantic_branch` resolves (donor compare, configured branch, router, backend) into a semantic branch plus a source string. This is the source of truth for `print_config_summary` so the reader can see which sparse path is live.

## Block-mask construction on XLA

CUDA uses `BlockMask.from_kv_blocks(...)` to feed FlexAttention. On TPU there is no FlexAttention; the Pallas kernel takes selected block indices directly, plus an exact-mask contract as side metadata. The construction pipeline:

1. Run the indexer to get `selected_block_indices` of shape `[B, n_query_blocks, top_n]`.
2. `classify_selected_block_masks` splits into `valid_mask`, `full_mask`, `partial_mask` using `arange`, `clamp`, gather, compare. No `nonzero`, `unique`, or `.item()`.
3. `compute_block_doc_ranges(doc_ids, block_size, seq_len, n_blocks)` runs once per packed row; `(block_doc_min, block_doc_max)` is reused across heads.
4. Build `mask_contract = {window_size, local_window, doc_ids, valid_token_counts}`. Per-batch torch tensors are not part of cache keys; `_make_exact_mask_contract_static_key` strips them.
5. Pass selected indices and mask contract into the Pallas pipeline.

`_safe_n_blocks(T, block_size)` calls `torch._check(T % block_size == 0)` under `torch.compiler.is_compiling()` so Dynamo and the XLA tracer see a clean integer division instead of a symbolic ceil_div that would have triggered `CantSplit`. `_first_explicit_kwarg` introspects the Pallas phase signatures and refuses to assert "exact mask supported" unless `local_window`, `doc_ids`, and `valid_prefix` are real kwargs and not just absorbed by `**kwargs` (a `**kwargs` catch-all is not evidence the three-phase pipeline honours the contract).

## MXU-friendly block sizes for v6e

TPU v6e has 128x128 MXU tiles. Sparse attention performance depends on block sizes being integer multiples of MXU tiles and on the kernel actually getting a dense matmul out of them. Defaults: `query_tile_size=256`, `kv_tile_size=1024`, `selected_block_size=256`, `block_stride=64`, `compression_block_size=128`, `top_n=8`. These give the MXU a 256x1024 dense tile per selected block (well above the 128 minimum), keep compression-side scoring under 128 to fit in VMEM without pre-chunking, and keep routed work per query block in the same order as the dense work it replaces. `block_stride=64` is the score-grid resolution; finer paid in scoring time without changing routing.

The blockized CUDA path on H200 uses `dsa_moba_block_size = 128` to align with FlexAttention's Triton tile. We keep these as separate per-backend defaults; MXU and Triton sweet spots are not the same.

## The contract tests

"Is this run actually sparse?" is not visible from the loss curve; we have caught at least three regressions where the model trained fine but was running dense in disguise. The tests have three layers:

1. **Static contract tests** in sanitized packed-row schema tests, sanitized block-sparse semantics tests, sanitized sparse contract tests. For any (donor compare, configured branch, router, backend) tuple the resolver must produce the documented semantic branch and source string. New aliases must update the tests in the same diff.
2. **Runtime-shape contract tests** in sanitized H200 sparse validation smoke tests. Synthetic inputs run the sparse path and assert the BlockMask geometry matches the expected `kv_num_blocks` distribution. The test that caught the FLASH-backend BlockMask bug lives here: the old code passed only `partial_mask` to `kv_num_blocks`, dropping strictly-below-diagonal KV blocks. The new test asserts the full strict-past set survives BlockMask construction.
3. **TPU-backward contract tests** for clustered sparse: `_chunked_causal_vjp_fn` is exercised through a fake-JAX harness on short sequences, and the 32768 / 32769 boundary contract is pinned explicitly.

The donor-runtime-compare path is the regression net. `--donor-runtime-compare=fa4` runs the dense FA4 backend in shadow alongside the sparse path; any divergence beyond tolerance fails CI.

## The long-context win we keep

Numbers are qualitative because operating points move with libtpu nightlies, but the shape is stable. T=128K passes in roughly the time linear scaling predicts in the hierarchical scoring pipeline on v6e-8; the fused `top_k` + chunked causal mask path keeps memory bounded. T=256K compiles and runs after the fused-scoring + per-chunk `dynamic_slice_in_dim` rewrite (CHANGELOG `2026-02-26`); the previous 16 GiB `block_scores` tensor is gone, replaced by per-chunk `[B, chunk, K, top_n]` output with per-chunk causal masking and `approx_max_k`. All tests at 4K-64K continue to pass; we refused to land the 256K change until the lower-length regression suite was clean. Long-context training on v6e-16 with a deep MoE preset converges under the same loss curve as the dense baseline where dense is still feasible.

Wins that did not survive: hierarchical 256K on v6e-8 OOM'd at the dense fallback for baselines even though the sparse path fit, so we pinned 256K training to v6e-16 with the EP=4, TP=2, DP=2 mesh. The non-fused scoring pipeline is a fallback only; new recipes enter with `fuse_topk=True`.

## How it lands in deployment

Production ships:

- **The sparse contract surface** from the public sparse-contract sample, essentially unchanged. The semantic branch enum, mixed-document policy descriptors, and benchmark-path vocabulary become the source of truth for the run config so "sparse eval" and "sparse train" cannot be confused at config time.
- **The block-mask classifier** from the public block-sparse semantics sample, ported to deployment tensor types. Mixed-document policy and the conservative `full_mask` rule come verbatim.
- **A FP8-fused sparse DSA kernel** (the public DSA sparse-attention sample) replacing Megatron's `unfused_dsa_fn` with gather-scatter: gather K/V at top-k indices, compute scores only on top-k, causal mask on top-k positions, softmax over top-k, weighted-sum gather. Memory drops from a 7.0 GiB FP32 scores tensor to 28.7 MiB scores + ~3.7 GiB gathered K/V at deployment shape. With five DSA layers per pipeline stage the saving compounds.
- **TileLang sparse MLA kernels** under `tilelang_sparse_mla/` for the H200 forward and backward, with a top-k selector. The deployment answer where the POC's FlexAttention path used to live.

Rewritten: the Pallas clustered-sparse pipeline becomes a Megatron-shaped op rather than a JAX bridge. Kernel-config defaults (256/1024/256/64/128) are the starting point; recipes can override per preset. Dropped: donor-runtime-compare is a research path, not deployment. Feature-flagged: `dsa_semantic_branch` and `block_sparse_router` graduate from POC knobs to recipe-level flags with explicit gates, so a recipe cannot start in `auto` and resolve to an unexpected branch.

## Ablations and what we kept

From the CHANGELOG entries that anchor each decision:

- **Block size 64 -> 128 on the CUDA blockized path.** Default `dsa_moba_block_size` moved from 64 to 128 to align with FlexAttention's Triton tile. The TPU default stays at 256; MXU sweet spot is different.
- **`torch.unique()` XLA guard in the public sparse-attention sample.** Old path forced a host-device sync. Current path raises on XLA and uses the gathered fallback via `_sparse_attention_chunked()`. Same family of fixes removed `.item()` and `nonzero()` from routing helpers.
- **GatedAttention support** in `DeepSeekSparseAttention` and `ClusteredSparseAttention`. Learnable sigmoid gate after `c_proj` in `_finalize_sparse_output`, `_sparse_decode_moba`, and the clustered fallback. Initialised at 0.5 to start as a no-op.
- **Clustered sparse JAX init on TPU.** Two tests crashed when JAX tried to grab the device. `_import_jax_cpu()` forces `JAX_PLATFORMS=cpu` for numerical-only tests so they do not collide with a live TPU process.
- **FLASH backend BlockMask fix.** `_moba_flex_attention` and `_block_indexer_flex_attention` previously dropped strictly-below-diagonal KV blocks because they passed only `partial_mask` to `kv_num_blocks`. They now pass the full `valid_mask`. Seven tests in sanitized blockmask regression tests cover the regression.
- **Sparse contract labels.** `semantic_track_label` (`exact-token DSA`, `blockized sparse CUDA`, `dense/full FA4`) is surfaced in `print_config_summary`. The operator-visible answer to "what backend am I really running".
- **Hierarchical scoring at T=256K.** Fused causal mask + top_k inside the chunk loop of `compute_block_importance_hierarchical()`; eliminated the 16 GiB `block_scores` tensor; chunk_size floor 64 -> 16 for finer memory control. T=256K passes on v6e-8 in roughly 271s scoring, 269s pipeline.

## Production checklist

- Lock kernel-config tile sizes per recipe; do not enter `auto`.
- Pin `top_n` to fit within the per-chip HBM budget at target context.
- Surface `semantic_track_label` in the run printout; refuse start if it does not match the recipe.
- Keep `valid_token_counts` and `doc_ids` outside the static cache key; carry them as data in the mask contract.
- Run donor-runtime-compare in CI for any sparse change; gate on numerical agreement within tolerance.
- Verify `_safe_n_blocks` divisibility for the target sequence length; reject configs that force `ceil_div` under `torch.compile`.
- Pin libtpu and Pallas pipeline versions; canary every nightly through the contract test surface.
- Keep `JAX_PLATFORMS=cpu` for numerical-only clustered-sparse tests.
- Default long-context recipes to `fuse_topk=True`; reject the non-fused path.
- Validate gated-attention initial sigmoid is 0.5; train only after the first warmup window.

## Kernel choice snapshot

| Context length | Backend | Kernel | Mask style |
|----------------|---------|--------|------------|
| <= 16K | CUDA | FA3/FA4 | dense |
| 16K-64K dense | CUDA | FlexAttention | BlockMask |
| 64K+ dense | CUDA | FA4 + MoBA index | learned block KV |
| <= 16K | TPU v6e | Pallas dense | dense |
| 16K-64K | TPU v6e | Pallas block-sparse | static MXU-aligned tiles |
| 64K+ | TPU v6e | Pallas clustered-sparse | contract-tested fallback |

```python
# MXU-aligned tile sizes we use on v6e
BLOCK_Q = 128
BLOCK_KV = 128
assert BLOCK_Q % 128 == 0 and BLOCK_KV % 128 == 0, \
    "tiles must align with the v6e MXU to avoid dense fallback"
```

## References

- the public block-sparse semantics sample, the public sparse-attention sample, the public sparse-contract sample, the public clustered-sparse attention sample, the public attention-validity sample
- the public DSA sparse-attention sample, the public TileLang sparse-MLA forward sample, the public TileLang sparse-MLA backward sample, the public TileLang top-k selector sample
- the engineering changelog entries on FLASH BlockMask drop, hierarchical scoring at 256K, MoBA block size, GatedAttention rollout, and the XLA `unique`/`item` guards.
- [DeepSeek-V3.2 Technical Report — DeepSeek, arXiv:2512.02556]
- [MoBA: Mixture of Block Attention — Lu et al.]
- [Native Sparse Attention (NSA) — arXiv:2502.11089]
- [Pallas: A JAX Kernel Language — official JAX docs]
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs — Xu et al., 2021]
