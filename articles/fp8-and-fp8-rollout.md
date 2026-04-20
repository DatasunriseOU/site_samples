---
title: "FP8 in the training stack: what shipped and what we rolled back"
description: "An engineer's account of rolling FP8 through the training stack: DeepGEMM block-scaled GEMMs, torchao Float8Linear, TransformerEngine FP8-aware activation checkpointing, and the parts that looked good on paper but lost the benchmark."
date: "2026-04-18"
author: "David Gornshtein"
tags: ["fp8", "training", "deepgemm", "torchao", "transformer-engine", "H200", "GB10"]
---

# FP8 in the training stack: what shipped and what we rolled back

FP8 is the first precision step where the textbook answer and the measured answer disagree on this stack. On paper it is free throughput: one byte per element, twice the tensor-core flops on H200, a known recipe from DeepSeek-V3. In practice, for the deployed hybrid we train (Mamba-3 majority, a minority of MLA/DSA attention, a fat MoE tail), the honest sum of wins and losses has been much closer to zero than the marketing pitch suggests. This post walks through the three FP8 surfaces we actually touched in deployed code this quarter: DeepGEMM as an alternative block-scaled GEMM backend, `torchao.float8.Float8Linear` as the default dense path, and TransformerEngine's FP8-aware activation checkpointing. For each we say what we tried, what the numbers said, and which pieces we kept.

## Why we cared about FP8 at all

The target is the deployed hybrid at 4.73B parameters on 8x H200 single node, pipeline-parallel off by default, EP=1, data-parallel across all eight ranks. The steady-state baseline before any FP8 work was 158 TFLOP/s (about 16% MFU) in BF16, MBS=6, GBS=48. A `nsys` profile on the same config gave a clean compute breakdown: the Mamba/SSM scan kernels owned roughly 34.5% of GPU time, cuBLAS GEMMs about 29.9%, elementwise ops 10.7%, the DSA indexer around 4.2%, NCCL a little under 5%, and MoE permute another 4.5%. That breakdown is the single most important piece of context for the entire FP8 story: GEMMs are a minority of the compute budget. Any FP8 benefit compounds against 30% of the budget, not against the full step time. Whenever we ran the numbers and the FP8 path did not clear its own overhead against that one-third slice, we rolled it back.

## DeepGEMM: the block-scaled baseline we actually want

DeepGEMM is DeepSeek's JIT-compiled CUDA GEMM library targeting SM90 (H100/H200) and SM100 (B200/GB200). It gives us two things that matter: FP8 GEMMs with 1x128 block-wise tile scaling, and fused MoE grouped GEMMs in contiguous and masked layouts. The block scaling is the part that makes FP8 training numerically tractable at our scale. Per-tensor dynamic scaling collapses the whole activation into one amax; one large outlier in a 7168-element row kills the dynamic range for the other 7167. DeepGEMM's scheme stores a separate FP32 scale every 128 elements along K, so outliers are contained to a single 128-block. DeepSeek-V3 trained a 671B model end-to-end under essentially this recipe; it is sufficient.

Concretely, activations are quantized per-token per 128-K-block online; weights are quantized per-output-channel per 128-K-block and persisted alongside the FP8 tensor as `weight_scale_inv` with shape `[ceil(N/128), K/128]`. The kernel consumes 128-element K tiles, loads the corresponding row/column scale pair, computes the FP8 dot product into an FP32 accumulator, and multiplies by the outer product of the two scales before accumulating into the next tile. The core expression inside the inner loop is literally `accumulator += dot(a, b) * a_s[:, None] * b_s[None, :]`. On SM100, scales are packed as UE8M0 (power-of-2 exponents); on SM90, they are FP32. DeepGEMM's `transform_sf_into_required_layout` figures out which format the current SM wants and reshuffles accordingly.

The integration story was the straightforward part. Dense linear layers where `in_features` and `out_features` are both multiples of 128 get swapped; embeddings, norms, small gates, and the DSA indexer projections stay BF16. The bigger payoff is MoE: instead of looping per-expert with a separate kernel launch and padding each expert's capacity-factor bucket, `m_grouped_fp8_gemm_nt_contiguous` consumes a concatenated token buffer plus a `grouped_layout` tensor mapping each token to its expert and dispatches one fused kernel. That removes both the launch overhead and the capacity-factor padding waste from the current MoE expert path. The advertised performance envelope is 1.1x to 1.5x over cuBLASLt FP8, up to 1550 TFLOPS on H800-class silicon; the large-K layers (K=7168 is typical) fall squarely inside DeepGEMM's happy path.

What we kept: DeepGEMM is now the reference implementation for block-scaled FP8 and is live for the MoE grouped-GEMM path on the H200 node. What we are explicit about: it does not apply to GB10 (SM121 is below SM90), it is CUDA-only so the XLA path inherits nothing, and JIT compile-time cold-start is real (minutes on first touch, fast after). The last point matters enough that `DG_JIT_CACHE_DIR` should point at a persistent disk on any shared runner.

## torchao Float8Linear: the default dense path, with caveats

For dense Linear layers outside the MoE tail, `torchao.float8` is the path of least resistance. Its design is module-swap rather than hooks: `convert_to_float8_training()` replaces `nn.Linear` with `Float8Linear`, and the quantize-cast-matmul logic lives inside that module's forward and backward rather than being injected by global `saved_tensors_hooks`. The forward is dispatched through a custom `torch.autograd.Function` decorated `@torch._dynamo.allow_in_graph`, which matters because the compiler can then fuse scaling, casting, and `torch._scaled_mm` into one graph.

The part we adopted without hesitation is the rowwise + grad-weight-HP recipe. Tensorwise is fastest but uses one scalar scale per tensor; rowwise keeps a separate scale per row (or column depending on the GEMM) and trades a slower CUTLASS kernel for a meaningful accuracy improvement. `ROWWISE_WITH_GW_HP` keeps the grad-weight GEMM entirely in high precision while still running the output and grad-input GEMMs in FP8. For deep stacks with mixed block types (A/M/E blocks with very different gradient dynamics), that trade-off has paid off in stability more than once.

The part that bit us was initialisation ordering. torchao's FSDP2 FP8 all-gather only works if the `Float8Linear` conversion happens before TP, before regional `torch.compile`, and before `fully_shard`. When the conversion runs after those passes, the weight subclass is never threaded through the sharding plan, `torch.compile` sees plain `nn.Linear`, and the `enable_fsdp_float8_all_gather=True` flag silently turns into a no-op. The all-gather then moves BF16 bytes instead of FP8 bytes and the advertised 10-20% multi-GPU speedup evaporates. The fix was trivial once located (move `apply_fp8_training()` to the top of the build pipeline); the diagnostic path was not. The lesson stuck: FP8 conversion is not a decorator, it is a load-bearing order constraint.

We also enabled `precompute_float8_dynamic_scale_for_fsdp()` after each optimizer step. It batches the per-weight amax computation into a single all-reduce (MAX) across ranks for every `Float8Linear` weight at once. On a single GPU it barely shows up; on the 8-rank FSDP2 mesh it removes N per-layer synchronisation points per step and tightens step-time variance.

## TransformerEngine's FP8 checkpoint: the piece we did not build ourselves

Our original FP8 activation compression was pure PyTorch: a `saved_tensors_hooks` layer that intercepted every `save_for_backward`, computed amax, divided by `fp8_max`, cast to e4m3fn, and stored a scale. Every saved tensor went through the hook; for a 52-layer model that is thousands of pack/unpack calls per micro-step, no awareness of whether the tensor was already FP8, and no integration with TE's `fp8_meta` amax history.

Transformer Engine's distributed checkpoint path does four things our hooks did not. First, it snapshots the `FP8GlobalStateManager` before the forward and restores it at recompute time, so the TE layers see the same recipe state on the second pass. Second, it recognises `Float8Tensor` (and the TE 2.x `QuantizedTensor`) as already-quantized and saves the raw FP8 buffer plus scale directly, avoiding a BF16 round-trip. Third, it uses the CUDA RNG tracker to keep dropout patterns identical across forward and recompute under tensor parallelism. Fourth, with `distribute_saved_activations=True` it splits the saved activation tensor across TP ranks and all-gathers during backward.

The decision of when to use it is not optional. At the block level Megatron has a hard switch: if FP8 or FP4 is active, it routes checkpointing through Transformer Engine; otherwise it uses the stock tensor-parallel checkpoint path. That is not a tuning knob, it is a correctness requirement: vanilla `torch.utils.checkpoint` uses Python callbacks that CUDA graph capture refuses to trace. Our own collision with this was memorable. With a one CUDA-graph implementation and selective recompute on a BF16 path, graph capture crashed with `Checkpointing is not compatible with .grad()`. With FP8 and the Transformer Engine graph path, the same recompute scope worked because TE's checkpoint is graph-capture-aware. The only working combination for FP8 plus selective recompute plus CUDA-graph capture on this stack is hybrid FP8 format, Transformer Engine graph capture, and recompute scopes placed only on the hot submodules; everything else deadlocks at capture, silently loses FP8 state, or both.

We kept TE checkpoint for the A-blocks (MLA and DSA) where TE linears live, combined with selective attention recompute inside the block: QKV projection and output projection are checkpointed by TE, the core attention (FA3/SDPA) is selectively recomputed from Q/K/V because that is the cheapest part to redo, and RMSNorm plus MLP stay inside the same TE-checkpointed scope. For M-blocks (Mamba-3) and R-blocks (M2RNN) we do not use TE checkpoint; they are not TE modules and the FP8 benefits do not apply. The hooks-based compressor remains as a fallback when Transformer Engine is not installed.

## What we rolled back

Three FP8 paths looked sensible in isolation and failed under the measured compute breakdown.

**TE FP8 GEMMs across the whole model.** With MBS=6 GBS=48 no-CG, the BF16 baseline was 158 TFLOP/s at 112 GiB peak. The same config under `--fp8-format hybrid` was also 158 TFLOP/s at 112 GiB. Adding `--fp8-param-gather` saved 5 GiB by removing the duplicated BF16 master in the all-gather buffer but moved throughput by less than 1%. The reason is exactly the breakdown above: GEMMs are 30% of compute, amax overhead (quantize, dequantize, amax history update) happens on every GEMM, TE holds both BF16 and FP8 copies of the weight, and `--use-precision-aware-optimizer` is mutually exclusive with `--fp8-param-gather` because of an Int16-vs-FP32 dtype assertion. For a hybrid Mamba-majority model, full-model FP8 is net zero.

**FP8 Mamba-3 MIMO scan.** This one we actually wrote and measured. A full port of the Mamba MIMO forward kernel with all five GEMMs in e4m3fn, FP32 accumulators, and per-token amax scaling cross-compiled cleanly for SM90a and emitted WGMMA FP8 instructions. It still lost: 0.73x to 0.91x versus BF16 on GB10 (SM121), and a projected ~1.07x on H200 under WGMMA FP8. Root cause: the MIMO kernel is not GEMM-bound. Rotary, trap-scaling, the SEGSUM mask, the state update, the diagonal reduction, the Z gate and the D term together dominate the kernel, and the FP8 cast-before-GEMM overhead outpaces the GEMM speedup. The microbenchmark (1.66x on an isolated 2-GEMM scan loop) did not predict the full-kernel result. The branch is kept as an R&D artifact; FP8 SSM is a dead path for deployment.

**CUDA-graph backward capture with FP8.** FP8 + `--cuda-graph-impl transformer_engine` + recompute + MBS=6 gave a pre-capture throughput of 258 TFLOP/s on iter 3. After TE captured the backward on iter 5 the same config settled at 218 TFLOP/s, a 15% regression. The TE CG backward overhead exceeds the launch-overhead savings on this graph. We kept CG-compatible code paths (the `is_graph_capturing()` guards, the branchless clamp in the DSA indexer, the `Mamba3._apply` fp32-preserving override) because they are correctness fixes independent of FP8, but we use with CG backward disabled for this model.

## What actually ships

The steady-state configuration, measured reproducibly on 8x H200 single node at MBS=8 GBS=64 with selective recompute on the MoE activation, MLP, and MLA up-projection submodules, Liger cross-entropy on the MTP head, and the DSA indexer-loss coefficient set to zero, is **265 TFLOP/s (27% MFU)** in BF16 with 110 GiB peak. Flipping to `--fp8-recipe tensorwise` on top of the same configuration nudges it to **269 TFLOP/s**; the win is small but non-negative, unlike `--fp8-recipe delayed`, which the transformer configuration layer explicitly rejects alongside MoE-activation recompute. For MoE-only FP8 with DSA coexistence, the selective patch that restricts FP8 context to MoE layers works cleanly after removing a stray `query.dequantize() + key.dequantize()` pair that had drifted into the installed DSA path and silently killed the zero-copy Float8Tensor route. DeepGEMM is live for the MoE grouped GEMM. TE checkpoint owns the A-block recompute. torchao Float8Linear handles the dense tail with `ROWWISE_WITH_GW_HP` and FSDP2 FP8 all-gather enabled. Everything else is BF16.

The part of FP8 most worth keeping is the discipline it forced on the rest of the stack: a compute breakdown we actually trust, a CG-capture story with named guards, a selective-recompute plan that matches the block type, and a hard rule that precision changes do not ship without a measured step-time delta on the full model. The part most worth rolling back is the assumption that FP8 is free.

## What ships, what we rolled back

| Path | Status | Reason |
|---|---|---|
| torchao Float8Linear (dense tail) | ships | `ROWWISE_WITH_GW_HP` + FSDP2 FP8 all-gather |
| DeepGEMM block-scaled MoE GEMM | ships | grouped FP8 GEMM with FP32 accum |
| TE FP8 activation checkpoint (A-blocks) | ships | CG-aware, snapshots `FP8GlobalStateManager` |
| Full-model TE FP8 GEMMs | rolled back | net zero on hybrid Mamba-majority model |
| FP8 Mamba-3 MIMO scan | rolled back | not GEMM-bound; 0.73-0.91x BF16 on GB10 |
| CG backward capture + FP8 | rolled back | 15% regression vs pre-capture |

```bash
# Steady-state preset: BF16 with selective FP8 surfaces.
train.py --fp8-recipe tensorwise \
         --recompute-modules moe_act,mlp,mla_up_proj \
         --cuda-graph-impl transformer_engine \
         --no-cuda-graph-backward
```

## References

- Public FP8 rollout notes in the public engineering project
- Public precision and backend comparison notes in `MegaCpp sample pack`
- Public training review summaries attached to the public engineering documentation
