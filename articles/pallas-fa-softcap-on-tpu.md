---
title: "Pallas FlashAttention with logit softcap on TPU v6e"
description: "The softcap-plus-local-window Pallas kernel we ship on v6e, how it integrates with our document-masking segment IDs, and the MFU we keep."
date: "2026-04-18"
tags: ["pallas", "tpu", "v6e", "flash-attention", "softcap", "doc-masking"]
---

Gemma-style logit softcap is the one attention variant that breaks every "just use the stock kernel" story. The CUDA side gets FA3 with a native `softcap=` parameter; the TPU side does not. On v6e we ship our own Pallas FlashAttention with fused softcap, local window, block-sparse grid shrinking, and segment-ids-based document masking. This post is specifically about that kernel — distinct from the broader [pallas-on-tpu](./pallas-on-tpu.md) survey — and what its MFU looks like once the dust settles.

## Why MegaCpp cares about this

Our production configs train with `attn_softcap=30` and keep `output_softcap=15` separately on the logits. Softcap is a stability lever: `s = softcap * tanh(s / softcap)` squashes outlier attention logits before softmax and meaningfully reduces loss spikes during long training on hybrid stacks. It is cheap on paper and expensive in practice if you implement it as a separate HLO op that walks the attention score matrix once more than it needs to. On v6e that extra pass is not free because the score tensor is VMEM-sized and the extra roundtrip eats HBM bandwidth that the MXU is waiting on ([Gemma 2 — Google, 2024](https://blog.google/technology/developers/google-gemma-2/)).

The other reason: our document masking is packed-sequence `segment_ids` on both the Mamba and attention sides. Any attention kernel we ship on v6e has to consume `segment_ids` directly; we do not want a dense mask materialised at context lengths of 16K and up. Combine softcap with segment-ids with a local-window option and grid shrinking, and you end up needing a custom Pallas kernel whether you wanted one or not.

## What we built in the POC

the public Pallas softcap kernel sample is the whole thing. It is a ~2600-line module based on `jax.experimental.pallas.ops.tpu.flash_attention` with four modifications and one escape hatch. The escape hatch first: when `softcap=0`, `local_window=0`, and no composed mask is requested, the public API delegates straight to `torch_xla.experimental.custom_kernel.flash_attention`. That is the stock native Pallas FA, and we have no reason to run through our code path when no softcap is in play.

The public entry point, `pallas_flash_attention_softcap`, is a PyTorch `autograd.Function`. Forward and all three backward kernels (dQ, dK, dV) are traced through `torch_xla.experimental.custom_kernel.trace_pallas`. This is a native-XLA path: no `call_jax` bridge, no 2.5x overhead tax that the legacy Splash fallback pays. The four modifications:

1. Softcap is fused after `sm_scale`, directly inside the attention inner loop. The kernel computes scores, scales by `sm_scale`, applies `s = softcap * tanh(s / softcap)`, then feeds softmax. Fused softcap means we do not re-read the score matrix and do not materialise a softcapped intermediate.
2. Local window is handled two ways at once. At grid-construction time we skip KV blocks that lie entirely outside the window for their Q block, which is the Splash-style optimisation. Inside the boundary blocks we apply a per-element mask. Memory footprint is O(T*W), not O(T^2).
3. Block-sparse via `scalar_prefetch`. The kernel consumes a three-valued `block_mask` — 0 skip, 1 partial, 2 full — computed on the host and dispatched through `scalar_prefetch`. A full block skips mask application entirely. A partial block uses a precomputed dense mask from `_build_partial_mask_blocks`, keyed by block-shape bytes so repeated partial masks deduplicate. The grid itself shrinks so that empty rows of KV blocks are not iterated at all.
4. Segment-ids document masking sits alongside the block mask. `q_segment_ids_tile_ref` and `kv_segment_ids_tile_ref` are tiled per batch and applied on every active block, because segment boundaries are not captured by the block mask. The equality check `q_segment_ids == kv_segment_ids` is ANDed into the mask inside the kernel. For full blocks (block_mask==2) the base mask is known all-True, so we only OR in the segment-ids check.

Mask composition is a separate surface. `AttentionMask` is a pure-Python/NumPy object with `CausalMask`, `LocalMask`, `FullMask`, `ChunkedCausalMask`, and `SinkMask` subclasses, and `&`/`|` operators to compose them. The `_build_mask_from_flags(causal, local_window)` cache builds the mask lazily from legacy flags. `build_block_mask(q_len, kv_len, block_q, block_k)` walks the mask and emits the int32 block-mask array. Running this on the host, not inside JIT, keeps the kernel's scalar_prefetch inputs static — which matters because a dynamic block mask would force a recompile per shape.

Default block sizes are `block_q = block_k = block_k_major = 512` across the Q, dKV, and dQ passes. These are the numbers that fit inside our VMEM scratch budget on v6e at our head dim and do not spill. `_active_block_sizes` is a module-level dict you can swap before warmup, deliberately not inside the training step, because swapping it mid-training would blow the kernel cache and trigger recompile.

GQA and MQA are handled via batch-fold. When `num_kv_heads < num_heads`, `_native_fa_with_gqa_reshape` reshapes the batch dimension so the native FA kernel sees the contract it expects (`(B*H_kv, gqa_groups, T, D)` for Q, `(B*H_kv, 1, T, D)` for K/V). This is materially cheaper than `repeat_interleave(...)` on K/V and avoids materialising the expanded K/V copies in HBM. Segment ids are reshaped alongside to the folded batch layout, and the explicit XLA sharding annotations are skipped when GQA is active because the old partition spec no longer matches. This is the fix that landed in March for the `xla_flash_gqa_uses_batch_fold_not_repeat_interleave` regression.

Backends live behind `--xla_flash_attn` and `--splash_attn` in the TPU attention dispatch layer; both route through the public Pallas softcap kernel sample on TPU. The legacy `call_jax` Splash path is only used when the trace-pallas kernel is genuinely unavailable (old torch_xla, missing JAX, etc.). `enable_xla_flash_attention(attn_softcap)` and `enable_splash_attention(attn_softcap)` both end up setting `_xla_flash_softcap_fn = pallas_flash_attention_softcap` so downstream dispatch sees one kernel.

the public attention-validity sample is the contract that says what "valid prefix" means for a row: `"none"`, `"token_prefix"`, or `"slot_prefix"` modes, plus optional auxiliary `token_prefix` for trimming a partial slot. The Pallas softcap kernel does not look at this directly; the attention caller in the main model runtime module does and converts validity into the right segment-ids layout or block-sparse frontier before invoking the kernel. Keeping the validity contract normalised outside the kernel is deliberate — we do not want the kernel recompiled every time the call site changes how prefixes are expressed.

## How it lands in production

Lifted as-is: the public Pallas softcap kernel sample, the mask composition classes, the splash-info caches, and the fused softcap-plus-segment-ids inner loop. This is not Megatron-shaped code; it is a kernel we invoke from inside the Megatron attention module. The adapter lives in the Megatron attention spec and calls the same `pallas_flash_attention_softcap` entry point.

Rewritten: the TPU attention dispatch layer. The production side picks a smaller surface than the POC — CUDA gets FA3 with native softcap, TPU gets the Pallas softcap kernel, and there is no FlexAttention fallback in production. FlexAttention stays in the POC for experiments. Less surface, fewer bugs.

Dropped: the `call_jax` Splash fallback. On the production target (torch_xla nightly aligned with our custom build) the trace-pallas kernel is always available. The fallback is a liability that keeps complaints about `call_jax` latency alive; we delete it for production and keep it for historical TPU generations in experiments.

Moving to a kernel path: the fused softcap epilogue for the `output_softcap=15` final-logit step. That one is not in the attention kernel; it is the CCE/Liger path on CUDA and an XLA fusion on TPU. We leave that alone here.

Feature flags in production: `attn_softcap`, `local_window`, and a choice between dense block mask and the block-sparse scalar-prefetch path. `block_q`/`block_k` are not user-tunable at runtime; they are pinned to 512 with a calibration override.

## Ablations and what we kept

The kernel surface, summarised:

| Feature | POC | production | Notes |
|---------|-----|--------------------|-------|
| Softcap fused after `sm_scale` | yes | yes | One pass over the score tile |
| Local window via grid shrink | yes | yes | O(T*W), not O(T^2) |
| Block-sparse via `scalar_prefetch` | yes | yes | Host-built int32 block mask |
| Segment-ids document masking | yes | yes | ANDed inside the kernel |
| GQA via batch-fold | yes | yes | Replaces `repeat_interleave` |
| `call_jax` Splash fallback | yes | dropped | Trace-pallas only in prod |
| Pallas RMSNorm epilogue | tried | dropped | XLA lowering was within ~1-2% |
| Softcap+dropout fused | tried | dropped | Cardinality churn not worth it |

The CHANGELOG history for this feature is dense. The lessons that survived contact with real chips:

1. Softcap and causal fold together. Two separate HLO passes for tanh and for softmax walked the score matrix twice; one pass is materially cheaper on v6e because HBM bandwidth is the bottleneck, not MXU.
2. Block sizes of 512 across all Q/dKV/dQ passes hit the sweet spot. 256 left VMEM underutilised; 1024 spilled and cost more than it saved. The number is not magical, it is the first multiple of `NUM_LANES * NUM_SUBLANES` that fits our head dim without spill.
3. GQA via batch-fold beats GQA via `repeat_interleave`. The `_native_fa_with_gqa_reshape` path landed after a P1 bug where the native XLA FA call was silently materialising the expanded K/V copies. Backward compat for segment ids in the folded layout was the subtle part.
4. Per-call softcap overrides used to be ignored. Splash and Pallas FA read a global softcap instead of the per-call argument at three sites. The regression test `_assert_flash_attention_per_call_softcap_overrides_global` was added after that.
5. Gate-bias vs softcap ordering is consistent across backends. FlexAttention applies softcap first, then gate_log bias; chunked fallback does the same. We audit this in a test because a hidden reorder would be silent.
6. The `Splash` GQA path fails fast on invalid `H_q % H_kv != 0` geometry. It used to silently rely on reshape arithmetic; now it raises.
7. Unbounded kernel caches in the TPU attention dispatch layer were a memory leak under long jobs. The caches are now keyed by a tuple of the shape/mask inputs and bounded by size, not by process lifetime.
8. Document-boundary correctness under GQA batch-fold required reshaping segment ids with the same layout. There is a regression test specifically for `native_fa_with_gqa_reshape_folds_batch_and_replicates_segment_ids`.
9. MFU on a depth-52 hybrid preset with `attn_softcap=30`, `local_window` off, `block_q=block_k=512` lands in a band we are comfortable keeping. We do not publish hard numbers because we are still closing the gap to Trillium-reference FA runs, but the number we hold steady is not inside "uh-oh" territory; it is inside "we ship this" territory.

The negative lessons — features we tried and removed:

- A dynamic `local_window` per step. Bad idea; every change triggered a recompile. We pin at config time.
- A Pallas RMSNorm fused into the attention epilogue. XLA lowering was inside a percent or two; not worth a second kernel.
- A softcap-and-dropout fused kernel. Dropout cardinality changed too often during ablations; the extra kernel was a maintenance cost we could not justify.

## Production checklist

Minimum invocation surface from the Megatron attention spec:

```python
from pallas_fa_softcap import pallas_flash_attention_softcap, build_block_mask

block_mask = build_block_mask(q_len, kv_len, block_q=512, block_k=512)
out = pallas_flash_attention_softcap(
    q, k, v,
    sm_scale=sm_scale,
    softcap=30.0,            # per-call, never read from a global
    local_window=0,          # pinned at config time, not per step
    block_mask=block_mask,   # int32, host-built
    q_segment_ids=q_seg,     # packed-sequence doc masking
    kv_segment_ids=kv_seg,
)
```

- All TPU training uses `--xla_flash_attn` (trace-pallas). `--splash_attn` routes through the same kernel and is kept as a compatibility alias, not an independent path.
- `call_jax` Splash fallback is forbidden in production; a startup assert fails the run if the trace-pallas loader did not succeed.
- `attn_softcap` is a per-call argument, not a global; regression tests guard the override at every backend dispatch site.
- `block_q = block_k = 512` at our head dim on v6e; changes go through calibration.
- GQA uses batch-fold, not `repeat_interleave`; the regression test must pass for any shape change.
- Document masking uses `segment_ids` end to end; no kernel may accept a dense doc mask in production.
- Mask composition (`CausalMask`, `LocalMask`, `FullMask`, `ChunkedCausalMask`, `SinkMask`) is pure Python/NumPy and imports cleanly on CUDA-only systems.
- the public attention-validity sample normalisation runs outside the kernel; prefix shape changes do not touch the kernel cache.
- Kernel caches are bounded and keyed by shape/mask; no unbounded growth under long jobs.
- Gate-bias vs softcap ordering is audited by a cross-backend test; do not reorder.

## References

the public Pallas softcap kernel sample, the TPU attention dispatch layer, the public attention-validity sample, the main model runtime module, the public sparse-attention sample, the public Mamba compile-wrapper sample, the public backend-matrix sample. External: [FlashAttention-2 — Dao, 2023](https://arxiv.org/abs/2307.08691), [Gemma 2 — Google, 2024](https://blog.google/technology/developers/google-gemma-2/), [jax.experimental.pallas — JAX docs](https://docs.jax.dev/en/latest/pallas/index.html), [torch_xla custom kernels — PyTorch/XLA docs](https://docs.pytorch.org/xla/), [Splash Attention — JAX TPU kernels](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention).
