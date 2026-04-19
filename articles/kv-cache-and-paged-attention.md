---
title: "KV Cache and Paged Attention for the MegaCpp Specialist Ensemble"
description: "Per-specialist KV cache layout, MLA cache after weight absorption, paged attention adoption status, and what changes between H200 and GB10 - including the MegaCpp serving plan."
date: "2026-04-18"
tags: ["kv-cache", "mla", "paged-attention", "fa3", "h200", "gb10", "serving"]
---

MegaCpp serves eight specialist SLMs behind a router, each holding its own KV cache. The dominant memory line at decode is cache, not weights: per specialist, per request, per token, per layer. This post covers the cache layout we ship, what MLA's compressed latent buys versus standard FA3, where the paged-KV path stands, and what the MegaCpp serving plan does differently on H200 vs GB10.

## Why MegaCpp cares about this

Training runs causal attention without a cache, so cost is zero. At serving the picture inverts. Standard FA3 cache for one specialist scales as `2 * B * T_max * H * head_dim * bpe` per attention layer; for the depth-52 hybrid with 13 attention layers, batch=8, T_max=8192, H=24, head_dim=128, bf16, that is ~6 GB per specialist. Eight specialists co-resident wants the better part of an H200. Two things change the picture. The hybrid pattern: only attention layers carry a KV cache; Mamba layers carry an SSM state cache that is `O(d_state)` per layer, not `O(T_max * H * head_dim)`. MLA: the compressed-latent cache replaces full per-head K and V with one `kv_lora_rank`-wide latent plus a small RoPE'd key fragment. Paged attention sits on top as the substrate for shared block pools with prefix sharing.

## What we built in the MegaCpp training stack

Three KV cache implementations live in the MegaCpp training stack.

The contiguous FA3 cache layout uses tensors shaped `(n_cache_slots, B, T_max, H, head_dim)` for K and V (FA3-style: time before heads). Position is tracked per batch element via a `cache_seqlens` int32 tensor that `flash_attn_with_kvcache` updates in place. `attn_layer_map` maps global layer index to cache slot index, allocating slots only for attention layers in a hybrid pattern. On depth-52 with 13 attention layers, this drops cache memory by ~75% vs "one slot per layer".

The MLA KV cache is the compressed-latent layout. Per attention layer it stores `low_rank_caches[layer_idx]` of shape `(B, T_max, kv_lora_rank)` and `rope_caches[layer_idx]` of shape `(B, T_max, 1, qk_rope_head_dim)`. With `kv_lora_rank=512` and `qk_rope_head_dim=64`, each cached token costs `2 * (512 + 64) = 1152` bytes per layer in bf16, vs `2 * 24 * 128 * 2 = 12,288` bytes per layer for standard MHA - ~10x compression before any quantisation.

The quantized KV variant carries two methods: PolarQuant (Han et al., AISTATS 2026) preconditions with a random orthogonal matrix, applies a recursive polar transformation, and quantises angles whose analytic distribution avoids per-vector scale/zero-point overhead; TurboQuant (Zandieh et al., ICLR 2026) is the simpler successor with random orthogonal rotation plus per-coordinate scalar quantisation on normalised vectors. Both achieve ~3.8x compression at 4 bits with minimal quality loss. Inference-only (non-differentiable quantiser); 4-bit pack/unpack uses interleaved nibbles.

`MLAKVCache` is what the production serving plan picks. It composes with `attn_layer_map`. It does not compose with `QuantizedKVCache`: the polar/turbo quantisers operate on full per-head K and V, not on the compressed latent. We tested fitting a per-coordinate quantiser on the latent and the 4-bit quality loss was too large because the latent has higher per-coordinate variance than post-up-projection K and V. Dropped.

The KV update path is duck-typed. `MLA.forward` checks `_is_mla_cache = isinstance(kv_cache, MLAKVCache)`; on True it stores `(low_rank_main, key_rope)` via `kv_cache.update(self.layer_idx, ...)` and reads back via `get_layer_cache`. The standard FA3 path uses `flash_attn_with_kvcache` with `cache_seqlens` and the layer's `(k_cache, v_cache)` slot. Both advance via `kv_cache.advance(T_new)` after the last layer.

## Paged attention: what is wired, what is deferred

The paged path lives under a paged-KV block manager and adapter pair. The block pool is `(num_blocks, n_cache_layers, block_size, num_heads, head_dim)` for both K and V, allocated once at engine startup. Sequences hold a list of block indices; the adapter materialises a `(B, max_blocks_per_seq)` int32 `block_table` that FA3's paged-KV decode path reads directly. Free blocks live in a LIFO list for O(1) alloc/free; a prefix-cache dictionary maps content-based keys to block indices, with reference counting and LRU eviction for back-pressure.

Two adapter modes. When `block_manager.block_size % 256 == 0` (FA3's hard requirement on `page_block_size`), `_fa3_paged_ok` is True and `CausalSelfAttention.forward` calls `get_pool_layer_cache()` to pass the pool tensors plus `block_table` straight into `flash_attn_with_kvcache` - zero-copy decode. Below 256 the adapter falls back to `get_layer_cache()` which gathers into a temporary buffer; correct, no longer zero-copy.

Adoption status: the paged-KV block manager is **structurally complete but DEFERRED from production**. It emits an INFO log on instantiation pointing at the roadmap. The contiguous FA3 path is the production substrate today. Paged-KV serving is wired through the model-level paged adapter contract with explicit runtime truth, but execute-and-prove proof for the scheduler-managed paged decode lane is still open.

Constraints we hit: `page_block_size` must be a multiple of 256 for FA3 zero-copy - we default to 256 because smaller blocks (16, 64) made prefix sharing finer-grained but lost the fast path, and the fall-back gather is a measurable hit at high QPS. Prefix cache identity is content-based (hash of token IDs plus an `adapter_key`), so the same prompt across different specialists or LoRA bundles is not mistakenly aliased. Diagnostic counters (`_diag_block_reads`, `_diag_block_writes`, `_diag_advances`) are non-optional - we added them after the "allocated but never connected" failure mode where the block manager allocated blocks but a wiring bug routed the block table to the wrong adapter, leaving the model's attention forward never reading or writing them.

## Per-specialist budget

For one specialist on one H200 device - depth 52, 13 attention layers, MLA with `kv_lora_rank=512`, `qk_rope_head_dim=64`, bf16 cache, T_max=8192:

Per-token MLA cost per layer: `(512 + 64) * 2 = 1152` bytes. Across 13 attention layers per token: ~15 KB. Per sequence at T_max=8192: ~120 MB. At batch=8 concurrent serving: just under 1 GB per specialist. Eight specialists co-resident: ~7-8 GB cache for the whole ensemble at full T_max and batch=8. Fits alongside FP8 weights and the working set on one H200.

Same shape with standard MHA: `2 * 24 * 128 * 2 = 12,288` bytes per token per layer, ~160 KB per token across 13 attention layers, ~1.3 GB per sequence at T_max=8192, ~10 GB per specialist at batch=8, ~80 GB across eight. Does not fit. This is what drove MLA into the production attention stack independently of any training argument.

The Mamba state cache (`MambaInferenceParams`, instantiated by `PagedKVCacheAdapter` when `has_mamba=True`) holds SSM state per Mamba layer at `O(d_state * n_groups * d_inner)`. Constant in `T_max`, a few MB per Mamba layer per request - negligible vs attention cache, but a real line. The DSA indexer K cache (`indexer_k_cache`) is a separate small cache for the DSA indexer's per-layer K tensor, on by default for DSA-enabled specialists.

## H200 vs GB10

Three things change between H200 (141 GB HBM3e per device, fast NVLink) and GB10 (DGX Spark, 128 GB unified, 273 GB/s, sm_121a, NVFP4 capable). Headroom: GB10 has more total memory per device but slower bandwidth, and unified memory means CPU and GPU contend for the same pool, so expandable-segment allocation is on by default. Attention backend: H200 runs FA3 (and FA4 in the bounded contiguous-KV lane) for prefill and decode; GB10 with sm_121a lacks the same FA3/FA4 fast-path coverage, so the bounded dense decode lane is the production target. Paged-KV decode is structurally available on GB10 with the same 256-multiple constraint. Precision: GB10's NVFP4 reaches roughly 11% MFU in the validated lane; we keep the cache in bf16 because the 4-bit quantized KV path has a known quality cliff on the MLA latent and NVFP4-on-cache is not yet validated. GB10 is bandwidth-bound (data loading 0.3% / forward 33% / backward 67%), so dropping cache precision matters more than dropping weight precision - the NVFP4-on-cache experiment is on the roadmap.

## How it lands in MegaCpp

The serving plan keeps the MLA KV cache as the per-specialist cache type, contiguous FA3 as today's substrate, and the paged path as the deferred-but-wired Track B. The eight-specialist ensemble runs co-resident on one multi-GPU node with one shared scheduler holding per-specialist state.

The production MLA helper in the Megatron stack keeps the same KV layout as the MegaCpp training stack. The fused MLA RoPE Triton kernel wraps latent compression, RoPE on the decoupled portion, and the attention dispatch into one kernel, removing intermediate tensors that would otherwise inflate per-step memory by 18.7 GB on the depth-56 R-variant. The fused MLA projection path handles the pre-attention projection. Both compose with the MLA KV cache.

The DSA path uses an absorbed-MLA variant with MQA layout (single-head absorbed key, all query heads share it). This is the only place in production where the absorbed reformulation runs, and it is decode-only. Training MLA stays on the standard expand-then-attend formulation because the decomposed score requires materialising a `(B, H, T, T)` attention-weight tensor that breaks Flash Attention. Separate sparse-MLA forward and backward kernels cover the H200 FP8 lane.

Production substrate: contiguous FA3 with one cache per request. Paged is the next milestone because the eight-specialist ensemble benefits structurally from a shared block pool with prefix sharing on the system prompt and the per-specialist instruction prefix. Until the scheduler-managed paged sparse decode lane has its execute-proof receipt, production traffic stays on contiguous FA3.

## Ablations and what we kept

`MLAKVCache` vs `KVCache`: ~10x compression on our shape, no quality regression on intrinsic evals. Lifted as production cache. `attn_layer_map` skipping Mamba layers: ~75% cache reduction on depth-52. Lifted. Block size 256 vs 16 vs 64: 256 wins because of FA3's requirement; smaller blocks gave finer prefix sharing but lost the zero-copy path. `QuantizedKVCache` 4-bit on standard cache: ~3.8x compression at minimal quality loss but does not compose with MLA - held as a fallback for non-MLA specialists. Per-coordinate quantiser on the MLA latent: rejected (variance too high). Adapter-keyed prefix cache: shipped. Paged-KV scheduler-managed decode: wired, deferred behind a feature flag until the execute-proof receipt closes.

## Production checklist

- One `MLAKVCache` per specialist; share `attn_layer_map` with the model so only attention layers allocate cache slots.
- `page_block_size = 256` if you turn paged on; below 256 the FA3 zero-copy path is unavailable.
- Adapter key required in the prefix cache key for any deployment with more than one LoRA bundle or more than one specialist sharing a block pool.
- `_diag_block_reads` / `_diag_block_writes` / `_diag_advances` exported as serving metrics; alert on `block_writes_per_step == 0` while requests are in flight (the wiring-bug signature).
- bf16 cache by default; do not enable `QuantizedKVCache` on MLA specialists.
- On GB10, set `PYTORCH_ALLOC_CONF=expandable_segments:True` to keep cache allocations from fragmenting against the unified pool.
- DSA-enabled specialists: budget for the `indexer_k_cache` line; it is small but constant.
- Hybrid (Mamba) specialists: confirm `MambaInferenceParams` is instantiated by the adapter (`has_mamba=True`); the silent regeneration mode is hard to detect without the counters.
- Decode path uses absorbed MLA only on the DSA decode lane; the standard decode path uses the same expand-then-attend MLA as training.
- Production substrate: contiguous FA3 today; paged FA3 is wired-but-deferred behind a feature flag.

## Per-specialist KV layout

| Specialist | Per-layer KV | Cache shape | Paged status |
|---|---|---|---|
| Dense reasoning | full `K`, `V` (GQA, kv_heads=8) | `(num_blocks, kv_heads, head_dim, block_size)` | paged on H200 + GB10 |
| Wide MoE | full `K`, `V` (GQA) | same as above per layer | paged, per-expert dispatch unchanged |
| MLA specialist | `c_kv` + 1 RoPE'd key row | `(num_blocks, kv_lora_rank + qk_rope_head_dim, block_size)` | paged after weight absorption |
| Mamba/M2RNN layers | recurrent state, no KV | `(num_blocks, state_dim)` | not paged; rolled per step |
| MTP heads | sibling KV slabs | shares main cache via index | paged with parent |

The block-table walk we use on H200:

```python
def lookup_kv(block_table, slot, layer):
    block_id = block_table[slot // BLOCK_SIZE]
    offset = slot % BLOCK_SIZE
    return kv_cache[layer, block_id, ..., offset]
```

## Public references

- [MegaCpp public repository](https://github.com/DatasunriseOU/cppmega)
- [Public sample pack](https://github.com/DatasunriseOU/site_samples)
- [DeepSeek-V2 Multi-Head Latent Attention - DeepSeek-AI 2024]
- [PolarQuant: Quantization-Friendly KV Cache Compression - Han et al., AISTATS 2026]
- [TurboQuant: Random-Rotation KV Cache Quantization - Zandieh et al., ICLR 2026]
- [Efficient Memory Management for LLM Serving with PagedAttention - Kwon et al., SOSP 2023]
- [FlashAttention-3 - Shah et al. 2024]
