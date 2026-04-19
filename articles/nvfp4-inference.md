---
title: "NVFP4 Inference for the MegaCpp SLM Ensemble"
description: "Why we train in FP16/BF16 and ship in NVFP4, what Blackwell and GB10 actually give us, and which kernels survive the trip from B200 to DGX Spark."
date: "2026-04-18"
tags: ["nvfp4", "blackwell", "gb10", "inference", "quantization", "cutlass"]
---

We train the SLM ensemble in FP16/BF16 on H200 and B200 and serve it in NVFP4 on Blackwell-class hardware. This post walks the inference path: how a BF16 master checkpoint becomes an NVFP4 deployment artifact, which Blackwell features the kernels actually rely on, and where GB10 (DGX Spark, sm_121a) silicon forces us to diverge from B200 (sm_100a). It is written for engineers who already know what `mma.sync` is and want to know what to build, not what to read.

## Why this matters

Inference dominates the operational cost of an eight-specialist ensemble, and Blackwell is the first generation where FP4 is a first-class tensor-core datatype. The temptation is to treat NVFP4 as a free 4x. It is not. The win is real but smaller than the spec sheet implies because Blackwell inference at these model sizes is bandwidth-bound, not compute-bound, and because the two Blackwell ISAs datacenter sm_100a and consumer-class sm_121a expose different subsets of the FP4 path. If we get the recipe and the kernel selection right we keep one checkpoint format across training and serving and reclaim most of the bandwidth headroom. If we get it wrong we ship a checkpoint that runs slower than BF16 on one of our two serving tiers and silently miscompiles on the other.

## 1. Why NVFP4 and not FP8 for inference

We spent a session proving that FP8 does not pay for itself during training of a hybrid stack (a Mamba3-heavy SSM/attention/MoE mix). The structural reason is unsurprising in hindsight: dense GEMMs are roughly a quarter of compute on this model, SSM scans are about as much again, and the elementwise tail eats another double-digit slice. TransformerEngine pays an amax quantize/dequantize on every GEMM and carries a BF16 master copy alongside FP8 weights, so the marginal compute saving is consumed by the bookkeeping. End-to-end TFLOP/s and MFU on H200x8 came out indistinguishable from BF16; smaller batches were marginally worse.

Inference inverts the cost model. There is no optimizer state, no master weights, no per-step amax history; the per-tensor scale is computed once at quantize time and baked in. With NVFP4 the weight footprint drops by roughly 4x relative to BF16, and on bandwidth-bound silicon (anything above batch one on Blackwell, in these shapes) that translates almost directly into tokens per second. NVFP4 wins for the same reason FP8 lost: the constant factors flip when you delete the backward pass.

### Where NVFP4 does not pay back

We also evaluated NVFP4 inside the DSA indexer as a research probe and walked away. The indexer linears are tiny — a 64-to-512, a 3584-to-64, a 3584-to-8 — three of four below the FP8 cuBLAS crossover, let alone the FP4 crossover. The inner score-compute is bandwidth-bound on a multi-GB FP32 accumulator that does not care about input dtype. FP8 already cost a low double-digit percent on the indexer linear forward in a B200 benchmark. The fix is to fuse the einsum, ReLU, weighted-sum, and topk into one kernel; the dtype is a second-order knob. The same logic applies to NVFP4: small GEMMs do not amortise the quantize tax. We use NVFP4 where it earns its keep — the bulk MoE expert GEMMs, the attention projections, and the dense FFN linears.

## 2. The quantization recipe

The training-to-inference handoff is mechanical and stays that way so we can move a checkpoint from H200 training onto a GB10 or B200 deployment target in one transformation. From a BF16 master checkpoint we produce an NVFP4 artifact with three pieces per quantized tensor:

1. NVFP4 weight blocks (4-bit elements, E2M1, packed two per byte) with a 16-element block size along the K dimension.
2. A per-block E4M3 scale (one FP8 byte per 16 NVFP4 elements), which the `kind::mxf4nvf4.block_scale` MMA consumes directly.
3. A per-tensor FP32 amax used to recompose the global scale at load time.

This is the NVIDIA-canonical NVFP4 layout. Staying canonical matters because it is what the CUTLASS `BlockScaledMmaOp` and the matching `mma.sync` family expect with no additional shuffling. Inventing a custom layout means losing the vendor kernels and maintaining your own; on a moving target like Blackwell that is not a fight worth picking. Activations are quantized on the fly to NVFP4 at the GEMM input boundary using a per-tile scale derived from a running calibration. The SSM state and the residual stream stay in BF16.

### What we deliberately do not quantize

- The SSM scan kernels (Mamba3 SSD and the M2RNN combined kernels). The available FP8/FP4 SSD variants regress accuracy without a throughput win at these serving sizes, and these are not GEMMs in the cuBLAS sense.
- LayerNorm and RMSNorm. BF16 with FP32 accumulator sits well below the tensor-core ceiling; there is nothing to gain.
- The MoE router GEMM. Too small and too sensitive — the routing decisions multiply downstream cost, so we keep it BF16.
- The LM head. One projection per token, accuracy-critical, not a hot path.

## 3. Blackwell features we actually use

B200 (sm_100a) and GB10 (sm_121a) are marketed under one Blackwell umbrella but are two different ISAs. The inference-relevant subset:

| Feature | B200 (sm_100a) | GB10 (sm_121a) |
|---|---|---|
| `mma.sync kind::mxf4nvf4.block_scale` | yes | yes (warp-level OMMA) |
| `tcgen05.*` family (mma/ld/st/alloc/cp) | yes | absent |
| Tensor Memory (TMEM) | 256 KiB/SM | absent |
| 2-SM UMMA, TMA multicast | yes | effectively absent (cluster cap 1) |
| Dynamic SMEM budget | ~232 KiB | ~99 KiB |
| Memory bandwidth | ~8 TB/s HBM3e | ~273 GB/s LPDDR5X |
| FlashAttention-4 cubins | yes | rejected at driver |

GB10's tensor cores are, in NVIDIA's own framing, closer to the GeForce-style MMA model with FP4 and FP8 bolted on; the RT and DLSS hardware took the die budget that would have gone to TMEM. Practically, every CUTLASS or FlashInfer kernel hard-coded against the `tcgen05`-coupled UTCOMMA path will not target GB10 and never will. The dead-paths list (FA4 on sm_121a, CuTe DSL `tcgen05`/UMMA, trtllm-gen FMHA, B200 tile configs reused on GB10) is enforced as kernel selection, not as a code-review note.

The shared-memory delta drives everything else. The B200 default 128x256x256 mainloop tiling overflows GB10 SMEM at compile time, so every kernel we ship for GB10 is re-tiled — typically 128x128x128 or 64x128x128 with a smaller pipeline depth — and the tile selection is part of the deployment artifact, not a runtime decision.

## 4. Kernel choices, by layer

The inference kernel mix is selected per layer class. The selection is intentionally narrow because the cost of carrying alternative paths is real and the test matrix grows multiplicatively.

### Dense FFN and attention projections

NVFP4 weights, BF16 or NVFP4 activations. On B200 we use the CUTLASS `BlockScaledMmaOp` path with `kind::mxf4nvf4.block_scale`, persistent scheduler, TMA bulk tensor loads, and `tcgen05`-coupled accumulation in TMEM. On GB10 we use the same block-scaled MMA but at warp level — the OMMA family is present on sm_121a; only the TMEM-coupled UTCOMMA variant is absent — with TMA loads in single-CTA form, swizzled SMEM, and the re-tiled mainloop.

### MoE expert grouped GEMMs

This is where NVFP4 pays the largest dividend, because the expert weight matrices are the bulk of the model parameters. On B200 the CUTLASS NVFP4 grouped GEMM is the right call. On GB10 we route through the cuBLAS 13.2 path where it is officially tuned for Spark, falling back to the in-tree CUTLASS kernel only when grouping shapes fall outside the cuBLAS heuristic. We do not use the TRT-LLM `nvfp4_gemm_cutlass` MoE path on anything older than TRT-LLM 1.3.0rc2 with CUTLASS 4.4.2; the older combination produced silent numerical corruption.

### Attention

On B200, FlashAttention-class kernels with TMEM and `tcgen05` are available and we use them. On GB10, FA4 is silicon-blocked and trtllm-gen FMHA has no SM12x cubins. The working path is BF16 CpAsync + TMA + FP8 inline-PTX from a CUTLASS PR, or PyTorch SDPA via the efficient-attention backend. SDPA matches FA2/FA3 throughput on GB10, and the source-build cost of FA2/FA3 buys nothing, so we ship SDPA on GB10 and FlashAttention-on-Blackwell on B200. The MLA path is the same shape — FlashInfer FA2 prefill/decode/MLA all work on GB10; the trtllm-gen MLA path does not.

### SSM (Mamba3 SSD, M2RNN, MIMO)

BF16 throughout, with the bias, D, and dt tensors preserved in FP32. The precautionary non-FP8/non-FP4 guards in spec wrappers were never load-bearing — TE's wrap of the parallel linears handles the FP32 tensors cleanly. For inference we keep the SSM kernels in BF16 because throughput is set by SSM scan latency, not by the surrounding linears, and quantizing the linears below BF16 inside the SSM block produces no measurable end-to-end win on these serving sizes.

## 5. What this means in numbers

Headline ceilings, from the public Blackwell whitepaper and dev-forum confirmations cross-checked into a capability checklist:

| Silicon | BF16 (TFLOPS) | FP8 (TFLOPS) | FP4 (TFLOPS) | Memory bandwidth |
|---|---|---|---|---|
| B200 (sm_100a) | ~2,250 | ~4,500 | ~9,000 | ~8 TB/s HBM3e |
| RTX 5090 (sm_120a) | ~210 | ~840 | ~1,680 | ~1.8 TB/s GDDR7 |
| GB10 / DGX Spark (sm_121a) | ~100 | ~200 | ~400 | ~273 GB/s LPDDR5X |

The B200-to-GB10 BF16 gap at the tensor-core ceiling is roughly an order of magnitude, and the memory-bandwidth gap is larger still. For inference on small-to-medium SLMs the bandwidth gap dominates: NVFP4 buys back roughly 4x weight-bandwidth pressure relative to BF16 on the same silicon, which is why a deployment target built around GB10 is viable for the MegaCpp ensemble at all. End-to-end we see the NVFP4 path delivering low single-digit factor speedups over BF16 at the GEMM level on GB10 — well short of the spec-sheet ratio, broadly consistent with the bandwidth wall — and a bigger jump on B200 where the tensor cores can actually be fed.

GB10 is explicitly not a training target; it is a serving and validation surface. Production training stays on H200 and B200. Production inference runs on whichever Blackwell tier matches the SLA: GB10 for low-rate, low-latency, edge-style deployments and the small-context specialists; B200 for the long-context specialists and any tier that has to absorb burst load.

## 6. Toolchain pins that matter

The Blackwell ecosystem moved fast enough through Q1 2026 that pins matter as much as algorithmic choices. The combination we currently ship inference against:

```toml
# inference toolchain pins (excerpted)
cuda        = "13.2"
cublas      = "13.2"
cutlass     = ">=4.4.2"
trtllm      = ">=1.3.0rc2"
flash_infer = "FA2/MLA on sm_121a; FA-Blackwell on sm_100a"
te_recipe   = "NVFP4BlockScaling, RHT disabled on sm_120/sm_121"
ptxas       = "system CUDA 13.0+ (Triton-bundled 12.8 cannot parse sm_121a)"
```

Two non-obvious traps. TransformerEngine's NVFP4 recipe defaults to a Random Hadamard Transform pre-scale that consistently dies on sm_120/sm_121 with `CUDA Error: invalid argument` once M crosses a low threshold; the fix is `disable_rht=True` in the recipe construction. And Triton ships its own `ptxas` (12.8) that does not know what `sm_121a` is — point Triton at the system CUDA 13.0+ `ptxas` via `TRITON_PTXAS_PATH` or every JIT will fail.

## What we kept and what we threw away

Kept: BF16/FP16 training, NVFP4 serving with per-block E4M3 scales and per-tensor FP32 amax, canonical NVIDIA layout, BF16 SSM kernels, BF16 norms and router and LM head, separate kernel selection per Blackwell ISA, RHT disabled on the GB10/RTX-50-class recipe, SDPA on GB10 attention, FlashAttention-on-Blackwell on B200.

Threw away: FP8 for training on this model shape, NVFP4 inside the DSA indexer, custom non-canonical NVFP4 layouts, B200 tile configs reused unchanged on GB10, FA4 on sm_121a, trtllm-gen FMHA on SM12x, the older TRT-LLM/CUTLASS combination that silently corrupted MoE NVFP4 outputs, and the assumption that one Blackwell kernel selection covers both ISAs.

The through-line is the one constraint Blackwell inference does not let you negotiate: bandwidth. Quantize where it relieves bandwidth, leave the rest in BF16, and pick kernels per ISA. Everything else — checkpoint format, deployment artifact, calibration, recipe — falls out of that.

## References

- [Precision recipe: FP16, BF16, FP8, NVFP4](https://github.com/DatasunriseOU/site_samples/blob/main/articles/precision-recipe-fp16-bf16-fp8-nvfp4.md)
- [Training on H200 eight-GPU machines](https://github.com/DatasunriseOU/site_samples/blob/main/articles/training-on-h200-eight-gpu.md)
- [NVIDIA Transformer Engine user guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [TensorRT quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [TensorRT-LLM installation and support matrix](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
