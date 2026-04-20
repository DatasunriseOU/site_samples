---
title: "A Memory-Budget Anatomy for One Specialist on H200:8"
description: "Line-by-line breakdown of weights, gradients, Muon+AdamW state, activations, KV cache, communication buffers, allocator overhead, and fragmentation for a single specialist trained on 8x H200, with the GB10 contrast."
date: "2026-04-18"
tags: ["memory", "H200", "GB10", "fp8", "nvfp4", "muon", "adamw", "training"]
---

This system is an ensemble of specialist SLMs trained independently. The unit of capacity planning is therefore not "the full ensemble" but "one specialist on one 8x H200 node". This post walks that unit's per-device HBM budget line by line: weights, gradients, Muon+AdamW state, activations under the current per-block recompute policy, KV cache during eval and serving, communication scratch, allocator overhead, and the fragmentation tail. Numbers are anchored in an analytical memory estimator and repeated H200/GB10 calibration runs.

## Why this matters

Each specialist is roughly the depth-52/56 hybrid shape with attention, Mamba-3, and MoE blocks interleaved. An 8x H200 node has 141 GB HBM3e per device and a fast NVLink fabric; GB10 has 128 GB unified memory and a slower interconnect. The same definition runs differently on the two, mostly in the activation and communication columns. Every recipe change (a new MoE expert count, a different AdamW/Muon split, FP16 to FP8 weights) reshuffles which line dominates, so the budget has to be legible. The pre-flight estimator exists so a launch does not spend most of an hour compiling only to OOM at step 0.

## What the estimator models

The memory estimator is the per-device model. Three things matter about its shape.

First, the hardware table. It carries usable HBM per chip, for example H200 at 141.0 GB and GB10 at 128.0 GB unified, and it tracks dtype byte widths such as bf16 (2), fp8_e4m3 (1), and nvfp4 (0.5). That lookup is what turns "drop weights to FP8" into a real number.

Second, the breakdown. The schema separates replicated parameters such as embeddings, norms, and routers; TP-sharded parameters such as Q/K/V/O, MLP, and Mamba projections; EP-sharded MoE expert banks; optional feature families such as Engram, DSA, mHC, MTP, NCP, n-gram hash, structure embeddings, and tree FFN; plus gradients, Muon state, AdamW state, activations, MoE routing, feature activations, overhead, and runtime reserve. Below 10% headroom the estimator emits a tight-fit warning because compile peaks eat what looks like comfortable margin.

Third, the categorisation. Every parameter group is tagged with a shard type such as replicated, TP-sharded, EP-sharded, or CPU-only, and an optimizer type such as Muon or AdamW. Shard type drives the per-device divisor; optimizer type drives the per-parameter byte multiplier. That tagging separates the Muon column from the AdamW column instead of producing one opaque optimizer number.

Per-line budget for one specialist on one H200 device. Ranges are illustrative; exact values depend on the preset.

Weights. For the depth-52 hybrid at our standard width, the per-device parameter footprint after TP=2 plus EP=2 sits in the low tens of GB at bf16. Routers, embeddings, and norms stay replicated; attention Q/K/V/O, MLP, Mamba projections, and shared experts are TP-sharded; routed experts are EP-sharded. `wte` and `lm_head` are each `vocab_size * D` and replicated. Dropping weights from BF16 to FP8 with `--fp8` halves this column for the matrices torchao wraps; `fp8_all_gather` also halves collective bandwidth. NVFP4 weights would cut another 2x but we do not train weights in NVFP4 - it is the activation/compute path on GB10 only.

Gradients. The sharded gradient term uses the same TP and EP divisors as weights, plus a data-parallel divisor when FSDP2 is on for non-replicated tensors. On CUDA eager this is materialised once per step; an XLA SPMD compiled step uses fused scratch instead, scaled by an implementation-specific scratch factor. On H200 with FSDP2 on, the gradient column ends up a few to low-double-digit GB.

Optimizer state. The Muon + AdamW hybrid earns its keep here. Matrix parameters (attention projections, MLP, Mamba projections, shared and routed experts) route to Muon, costing 2 bytes per parameter for the bf16 momentum buffer. Embeddings, the LM head, RMSNorm scales, the MoE router, and a small set of output projections rerouted on CUDA TP>1 cost 8 bytes per parameter for fp32 m+v. Muon-eligible params dominate parameter count, so the optimizer footprint comes out closer to "2x params" than the "8x params" you would expect from pure AdamW.

Activations. The estimator uses the Megatron-style `34 * B * T * D` bytes per layer in fp16 units, scaled by `bpe / 2`. At our typical microbatch and 4k sequence length on H200, full activations land in the tens of GB per device per microbatch. With per-block checkpointing ("ABlocks: full; EBlocks: selective expert-GEMM only; MBlocks: never; RBlocks: full") the column collapses by 30-40% versus full materialisation, with the largest single saving being ~44 GB across 22 EBlocks via selective expert-GEMM recompute on the depth-56 ~4.7B MoE preset.

KV cache. Zero during training because causal training recomputes instead of caching. At eval and serving, the cache follows the attention-layer map rather than allocating one slot per total layer, saving about 75% on a depth-52 stack with 13 attention layers. MLA collapses this further by storing a low-rank latent state plus a small RoPE-specific component per attention layer. With `kv_lora_rank=512` and `qk_rope_head_dim=64` that is roughly 10x smaller than a full FA3 cache.

Comm buffers. Distributed-optimizer all-gather and reduce-scatter are bucketed into bus-bandwidth-friendly sizes because under-padded buckets cause stalls under MoE reshard pressure. With FSDP2 ZeRO-3 plus TP and EP, in-flight collective buffers are a few GB per device on H200, smaller than the gradient column but never zero. MoE dispatch churn produces the classic "no leak, but reserved creeps" pattern visible in allocator retry counters. `--fp8_all_gather` halves both the bandwidth and the gather-buffer footprint.

MoE routing. `moe_routing_gb` is its own line. The estimator computes `C_e = ceil(1.25 * B * T * top_k / n_routed)` with the 1.25 capacity factor. With EP, each chip holds `n_routed / ep_degree` experts post-dispatch but router logits stay replicated. On 64-expert MoE under XLA SPMD logical-batch fusion, this produced the LLO 4 GB tensor crash (`[64, 122880, 1536]` = 12.1B elements) the estimator now warns on; H200 does not hit the same int32-element ceiling but the dispatch scratch is still real.

Allocator overhead. `MemoryEstimate.overhead_gb` defaults to 1.5 GB - PyTorch's caching allocator slack plus the CUDA driver context and cuBLAS/cuDNN workspace pools. Flat reserve, not a function of model size, because workspace pools saturate quickly.

Fragmentation. Runtime memory debugging surfaces counters the estimator cannot model, such as allocator retries and inactive split bytes. When these climb mid-run while the working set is steady, the binding constraint is fragmentation, not capacity. The fix is usually `PYTORCH_ALLOC_CONF=expandable_segments:True` or a bucket-dtype change that reduces allocator churn.

Runtime reserve. `runtime_reserved_gb` sits between the estimated total and the usable HBM cap. It holds headroom for compile-time spikes and for the cuBLAS handle, NCCL communicator, and Triton autotuner allocations that happen after the estimator runs.

Per-device HBM budget, one specialist on 8x H200, depth-52 hybrid:

| Column                     | Typical size   | Dominant driver                          |
|----------------------------|----------------|------------------------------------------|
| Weights (bf16)             | low tens of GB | TP=2, EP=2, routed experts EP-sharded    |
| Gradients                  | mid single digits - low teens GB | FSDP2 DP divisor        |
| Optimizer (Muon + AdamW)   | ~2x params     | bf16 momentum dominates                  |
| Activations (per block)    | tens of GB     | `34*B*T*D` scaled by recompute policy    |
| KV cache (train)           | 0              | no cache during training                 |
| KV cache (serve, MLA)      | ~10x smaller than FA3 | `kv_lora_rank=512`, `qk_rope=64` |
| Comm scratch + overhead    | 1.5 GB default | NCCL, cuBLAS, autotuner                  |
| Runtime reserve            | cap minus est  | compile spikes                           |

Muon vs AdamW byte counts per matrix parameter:

```python
MUON_BYTES_PER_PARAM  = 2   # bf16 momentum only
ADAMW_BYTES_PER_PARAM = 8   # fp32 m + fp32 v
# so a Muon-dominated stack costs roughly 2x params in optimizer state,
# not the ~8x you would expect from pure AdamW.
```

## How it lands in a production stack

The estimator and its categorisation transfer cleanly. Because the estimator is pure Python with no GPU dependency, the same analytical model can serve both research and production launchers. What changes is the control surface: selective recompute is exposed through a public recipe interface, and the per-block manual policy maps onto selective core-attention checkpointing plus dedicated Mamba recompute rules.ute extension covering non-`TransformerLayer` modules under `CPPMEGA_MAMBA_RECOMPUTE=1`. The estimator does not yet model Megatron-selective at the same fidelity, so we cross-check against the public memory-debug tool peak after the first hundred steps.

FP8 weights, FP8 activations, FP8 MoE experts, and FP8 all-gather stack together by default on H200 (`nemotron_presets.py`); first and last two layers stay BF16 for stability. Each FP8 toggle halves a specific row of the budget. The Muon + AdamW split is the same hybrid; the live optimizer reroutes a small set of output projections on CUDA TP>1. The MLA cache and paged KV adapter (`PagedKVCacheAdapter`, `PagedKVBlockManager`) are the production serving substrate; training does not use them.

## Ablations and what we kept

The lines that moved the budget the most on the depth-56 ~4.7B MoE preset: bf16 grad-reduce switch ~9.5 GB; selective expert-GEMM recompute ~44 GB across 22 EBlocks; Mamba conv+BC recompute ~6 GB across 13 MBlocks; M2RNN block checkpointing ~8.9 GB across 4 RBlocks; viewless output and norm recompute ~7 GB. Together those moved the preset from `dbs=2` OOM to `dbs=32 @ 112 GB` with 31 GB headroom on 8x H200.

Fused kernels: the MLA RoPE Triton kernel saved 18.7 GB peak on the depth-56 R-variant; the fused Mamba conv kernel (CUDA, replacing grouped conv1d) saved another 18.7 GB. Liger fused linear cross-entropy avoids the `B*T*V` logits materialisation - on GB10 with depth-20, B=32, T=2048, Triton backend plus the `capture_scalar_outputs=True` graph-break fix landed at 65,939 MB peak vs 68,595 MB, a 3.9% reduction at +17% throughput.

Tried and dropped: `skip_eblock_checkpointing` (brittle under specific MoE dispatch configs); full-block MBlock checkpointing (collides with the FP8 packed-doc Mamba lane's [-448, 448] activation bound); the auto-fit retry that swapped `dbs=8 + ckpt=on` for `dbs=4 + ckpt=off` on OOM (broke the regional-compile receipt; we reordered the score function to preserve the checkpointing state first).

## Production checklist

- Run `auto_fit` before every recipe change; record the selection reason in the launch log.
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` on modern high-memory accelerators when fragmentation is a concern.
- Watch `num_alloc_retries` and `inactive_split_bytes` in the first few hundred steps; a steady climb is a fragmentation signal, not a leak.
- Hold at least 10% headroom against `usable_hbm_gb`; tighter fits routinely OOM during compile, not at step 0.
- Keep the FP8 stack (weights, activations, MoE experts, all-gather) on by default on H200; first and last two layers stay BF16.
- Cross-check the estimator's activation column against measured peak memory after step 100; if they diverge by more than 10%, fix the estimator rather than silently raising headroom.
- Do not reuse XLA SPMD's logical-batch dispatch scratch model on CUDA; CUDA does not have the int32-element ceiling and the warning is not portable.

## References

- the public memory-estimation and calibration notes
- MLA, KV-quantization, FP8-activation, and CPU-offload components
- the public optimizer and runtime notes
- the public NAM56R recipe sample
- public change notes
- [Megatron-LM activation memory formula - Korthikanti et al., MLSys 2023]
- [DeepSeek-V2 Multi-Head Latent Attention - DeepSeek-AI 2024]
- [Muon: An optimizer for hidden layers in neural networks - Jordan et al. 2024]
