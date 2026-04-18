---
title: "The MegaCpp Precision Recipe: FP16, BF16, FP8 and NVFP4 in One Stack"
description: "How MegaCpp picks a numerical format per op, per device and per phase: FP16 only as a floor, BF16 as the steady state, FP8 in selected GEMMs, NVFP4 at inference, and the master-precision rules that make Muon and AdamW agree."
date: "2026-04-18"
tags: ["bf16", "fp16", "fp8", "nvfp4", "mixed-precision", "training", "inference"]
---

We use four numerical tiers in the MegaCpp stack and we use them deliberately. BF16 is the steady state for training compute; FP16 exists as a SM<80 floor for development boxes; FP8 is opt-in on a curated set of GEMMs; NVFP4 is the inference target on Blackwell. The trick is that the recipe is not a global flag - it is a per-op, per-device, per-phase contract, and the master-precision rules that hold it together are deliberately different for Muon and for AdamW. This post walks through what each tier actually buys us, where the rollout discipline lives in the code, and how the inference NVFP4 path maps cleanly back from the training master. It is intentionally distinct from `fp8-and-fp8-rollout` (which is the training-stack post-mortem) and `nvfp4-inference` (which is the Blackwell deployment post); this one is the precision contract itself.

## Why MegaCpp cares about this

A precision recipe is the single most leveraged decision in a training stack: get it right and a 4.7B-parameter hybrid trains in BF16 across a single H200 node and serves in NVFP4 on Blackwell with one checkpoint format; get it wrong and you ship a model that is silently lossy on either end. The hybrid we train is heterogeneous - Mamba-3 SSM blocks dominate compute, an MLA/DSA attention minority is bandwidth-sensitive, the MoE tail is a fat GEMM zoo, and the optimisers have very different precision requirements. A single global "use FP8" switch destroys at least one of those subsystems. The recipe below is per-surface.

The bandwidth-vs-compute mix matters too. Our profiles say SSM scan kernels own roughly a third of GPU time, cuBLAS GEMMs another third, with elementwise glue, the DSA indexer, NCCL and MoE permute splitting the rest. Any FP8 rollout compounds against the GEMM third only; any FP4 rollout at inference compounds against bandwidth, not compute. The recipe encodes that arithmetic.

## What we built in the POC

A small set of public training and optimizer modules carries almost all of the precision logic, with the optimizer pair in the MegaCpp codebase acting as the master-precision authority.

The public FP8 training entrypoint in MegaCpp is the training-side dispatch into torchao's `Float8Linear`. It is a module-swap design: nominated `nn.Linear` layers are converted in place, and the cast-scale-matmul-cast logic lives inside the swapped module's forward and backward. The conversion policy is explicit and conservative. We convert attention Q/K/V/proj, MLP `c_fc`/`c_gate`/`c_proj`, Mamba `in_proj` and `out_proj`, MoE expert linears (when expert weights are linear, not fused), the DSA indexer projections, and every MLA projection without exception (matching the DeepSeek-V3 pattern). We exclude embeddings, the LM head, all norms, MoD and MoE routers, the small Engram and mHC projections, anything Conv1d, and any layer whose dimensions are not divisible by sixteen (the FP8 hardware alignment requirement). The rule of thumb baked into the comments is "GEMMs with K, N > 4096 typically benefit"; everything below that threshold, or where the matmul is not the dominant cost, stays BF16. The conversion routine returns a counts dict so we can log how many layers actually swapped and grep for surprises.

The public FP8 optimizer layer is the COAT FP8 AdamW wrapper (NVlabs, ICLR 2025). Adam's `exp_avg` and `exp_avg_sq` are stored in FP8 with Dynamic Range Expansion; the kernel buys ~3.7x optimizer-state memory savings against standard AdamW. The wrapper detail that matters here is that the COAT kernel demands FP32 params, so the wrapper casts the param slice to FP32 for the optimiser step and casts back to BF16 after. That ~1ms-per-step overhead is paid in exchange for the memory headroom that lets us fit a larger model or a larger microbatch.

The public FP8 activation-checkpointing layer quantizes saved activations during recompute windows. It quantises saved activations to e4m3fn during the gradient checkpoint save and decompresses to BF16 during recompute. There are two backends: a native `saved_tensors_hooks` path that pack/unpack with fused Triton kernels when available (one launch instead of three on pack, one instead of two on unpack), and an opt-in COAT backend that uses NVlabs's mixed-granularity quantisation. The native path is the default; COAT is an env-flag opt-in for memory-pressed configs. Both honour a minimum tensor size (default 16K elements) below which quantising is more overhead than it is worth.

The public Transformer Engine checkpoint notes document the third FP8 surface: TransformerEngine's FP8-aware activation checkpointing for the layers that actually run their forward inside `fp8_autocast`. The point is that TE layers already produce `Float8Tensor` / `QuantizedTensor` objects with proper scaling metadata; our generic hook would dequantise to BF16 then re-quantise from scratch, throwing away both the bytes and the scale. TE's `checkpoint()` saves the FP8 buffer plus its scale directly and snapshots the `FP8GlobalStateManager` so the recompute pass sees the same FP8 context as the original forward. The decision rule is mechanical: when `config.fp8` or `config.fp4` is set, attention/MLP blocks use `te_checkpoint`; otherwise they use the native path. There is no middle ground; mixing them produces silently wrong gradients.

The public DeepGEMM study is the alternative GEMM backend comparison. DeepGEMM uses 1x128 block-wise tile scaling - per-token scales for activations, per-channel scales for weights, both at 128-element K granularity, with FP32 scales on H100/H200 and packed UE8M0 (power-of-two-only) scales on Blackwell. The kernel processes 128-element K tiles, multiplies the FP8 dot product by the outer product of the two scale vectors, accumulates in FP32. Weights must be K-divisible by 128, M alignment for grouped MoE GEMMs is 128 tokens per expert. This is the path we took seriously as an FP8 alternative to torchao for the MoE expert GEMMs specifically; the comparison post (`fp8-and-fp8-rollout`) records the verdict.

The master-precision contract lives in the public Muon and AdamW implementations in MegaCpp. Muon stores momentum in FP32 (the optimizer states for Muon plus AdamW combined sit at roughly 20 GB on the production preset); the lerp inside Muon was at one point cast to FP32 to chase a NaN that turned out to be the MoE compile bug, and that cast was reverted once the real cause landed. AdamW gets its `m` and `v` in FP32 by default and FP8 (via COAT) only when explicitly enabled. Both optimizers operate on a BF16 parameter view, with FP32 master grads when Megatron's `grad_reduce_in_fp32` is on - that flag is the linchpin: if reductions are already FP32 we skip a downcast in the optimizer paths. The rule that ate a week of debugging: the per-op promotion choice for Muon versus AdamW is not symmetric, do not apply a "one master precision" abstraction across both.

## How it lands in MegaCpp

The MegaCpp precision contract is intentionally narrower than the POC's surface area.

BF16 is the default training compute precision on every device that supports it. FP16 survives only as the SM<80 development-box floor (where Triton's BF16 codegen emits PTX that ptxas rejects - Triton issue #1941); on production hardware FP16 is not exposed.

FP8 stays opt-in and is rolled out per-surface. The torchao `Float8Linear` swap is the production default for dense MLP and attention projections meeting the alignment criteria. The COAT optimiser is opt-in and gated on memory pressure: it wins when we want a larger microbatch or a deeper preset, it loses on the optimiser-step latency tail when we don't. The FP8 activation checkpointing is wired in tandem with TE's FP8-aware `te_checkpoint` - the rule "if `config.fp8`, use TE checkpoint" is a hard precondition, not a hint. DeepGEMM is shipped as the MoE expert grouped GEMM backend on H200 and Blackwell where its per-tile scaling beats torchao's tensorwise path; on the dense MLP path torchao stays the default because module-swap composes cleanly with Inductor (the `matmul_with_hp_or_float8_args` autograd Function is `@torch._dynamo.allow_in_graph`, which is exactly what lets compile fuse cast and matmul).

NVFP4 is the inference target on Blackwell. The training master is BF16; the conversion to NVFP4 happens at quantisation time, with TransformerEngine's `NVFP4BlockScaling` recipe and `disable_rht=True` on consumer-class Blackwell silicon (RHT crashes on SM120/SM121 above M=32; `disable_rht` is required, not optional). WGrad stays BF16 in the recipe override - that is intentional, it is the gradient path that is most sensitive to the FP4 dynamic range. The mapping rule from training is: weights that lived in BF16 with FP32 master move to NVFP4 with per-block scales; FP8 GEMMs that participated in training translate cleanly because the per-tile scaling philosophy is the same; FP8 optimiser states do not propagate, they were a training-only memory trick.

We drop several POC switches in production: the `NO_COMPILE` workaround that existed only because of the MoE recompile bug; the standalone DeepGEMM "1D2D" kernel variant for dense GEMMs (kept only for MoE); the FP8 conversion of any layer with a sub-16-divisible dimension (the alignment check is now an assertion, not a warning).

## Ablations and what we kept

The CHANGELOG is unusually candid about FP8. The headline result on H200 is that FP8 is a net win on the dense MLP and attention projections that meet the alignment criteria, a wash on the MoE expert GEMMs unless DeepGEMM's grouped path is used, and a clear loss on small projections (Engram, mHC, MoD/MoE routers) where the cast overhead dominates the matmul savings. The conversion policy in `fp8_training.py` reflects that triage directly.

For NVFP4 the ablation that matters is on GB10 (DGX Spark, sm_121a), where the expected 8x compute speedup against BF16 collapses to roughly 1.2x to 1.4x because the silicon is memory-bandwidth-bound (273 GB/s) rather than compute-bound. The CHANGELOG captures the kernel-level numbers (4096-cube at 1.34x, 16384x4096-square at 1.36x). The implication for the recipe is honest: on GB10 we still ship NVFP4 because the memory footprint matters for the SLM ensemble and because we want one checkpoint format across both Blackwell tiers, but we do not promise an 8x serving speedup.

For Muon, the CHANGELOG records a sequence of NaN tests that all looked like Muon-precision bugs and were not. The "Muon BF16 compile NaN" was misdiagnosed; the real cause was the MoE `_overflow_total` Python-int counter triggering Dynamo recompiles. Once that was fixed, an 8-GPU DDP run with `torch.compile` and BF16 Muon trained cleanly. The lesson encoded into the recipe: do not chase precision casts when the symptom is recompile chatter; verify the compile cache first.

For AdamW, weight-decay alignment with the Moonlight paper (arXiv:2502.16982) was a precision-adjacent fix worth noting: a previous bug had Muon WD decaying to zero across the schedule and AdamW WD on RMSNorm gamma being zero by default. The fix landed `--adamw_weight_decay=0.1` and a non-decaying Muon WD, with the explicit per-step values logged in the CHANGELOG. Precision recipes interact with weight-decay schedules; we test both together.

## Production checklist

- BF16 is the only training compute precision exposed on production hardware. FP16 exists only on SM<80 dev boxes.
- The FP8 conversion policy in `fp8_training.py` is the source of truth for which layers swap; embeddings, LM head, norms, routers, Conv1d and any layer with a dimension not divisible by sixteen never swap.
- When `config.fp8` is set, all checkpointed attention and MLP blocks use `te_checkpoint`. Native checkpoint plus FP8 forward is a configuration error and is rejected at startup.
- COAT FP8 optimizer is opt-in and gated on memory pressure; the BF16-to-FP32-and-back cast in the optimizer step is the documented overhead.
- DeepGEMM is the MoE expert grouped GEMM backend on H200 and Blackwell; torchao stays the dense default for `@torch._dynamo.allow_in_graph` composability.
- Muon momentum is FP32; AdamW `m`/`v` are FP32 unless COAT is on; FP32 master-grad reductions (`grad_reduce_in_fp32`) skip the downcast in the optimizer paths.
- NVFP4 inference uses TE's `NVFP4BlockScaling` with `disable_rht=True` on SM120/SM121 and `override_linear_precision=(False, False, True)` so WGrad stays BF16.
- One checkpoint format from training to NVFP4 serving; the conversion is at quantisation time, not at training time.
- On GB10 the published serving speedup vs BF16 is the measured 1.2x to 1.4x range from the kernel benchmarks, not the silicon spec sheet's headline number.
- Precision changes always re-run the optimiser weight-decay regression tests; the Muon/AdamW WD interaction is not assumed stable across casts.

## References

- `fp8_training.py` - torchao `Float8Linear` swap policy
- `fp8_optimizer.py` - COAT FP8 AdamW wrapper, FP32 cast in optimizer step
- `fp8_activations.py` - FP8 activation checkpointing, fused Triton pack/unpack
- `fp8_te_checkpoint_design.md` - TE FP8-aware checkpoint design
- `fp8_deepgemm_notes.md` - DeepGEMM 1x128 block-scaled GEMM notes
- `muon.py`, `adamw.py` - master-precision and momentum-storage rules
- the main model runtime module - `NVFP4BlockScaling` recipe wiring with `disable_rht=True`
- CHANGELOG_GB10 entries on NVFP4 vs BF16 kernel speedups
- CHANGELOG entry on Muon weight-decay alignment with Moonlight
- [DeepSeek-V3 - DeepSeek, arXiv:2412.19437]
- [COAT - NVlabs, ICLR 2025]
- [Moonlight Muon WD - arXiv:2502.16982]
- [TransformerEngine NVFP4 issue #2372 - GitHub]
