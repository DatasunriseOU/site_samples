---
title: "Training the MegaCpp SLM Ensemble on GB10: A Grace Blackwell War Story"
description: "Honest field notes from bringing the MegaCpp SLM Ensemble up on NVIDIA's GB10 / DGX Spark — silicon surprises, NaN bisects that ate days, regressions caused by our own patches, and the software-stack choices that actually held."
date: "2026-04-18"
tags: ["gb10", "blackwell", "sm121a", "nvfp4", "tilelang", "transformer-engine", "training"]
---

The GB10 / DGX Spark looks, from the marketing slide, like a small Blackwell. It is not. It is a different ISA wearing the Blackwell logo, with a desktop-class die, a roughly 273 GB/s LPDDR5X memory bus, and a software stack that assumes you are running on a B200 until you prove otherwise. This post is the unvarnished account of bringing our hybrid SSM/attention/MoE stack up on this box: the silicon traps, the NaN hunts that turned out to be our own patches, and the software-stack recipe that finally held.

## Why this matters

GB10 is the first Blackwell-branded box most teams will actually touch with their own hands. Treating it as a small B200 is the failure mode that costs the most time. It shares the brand and the FP4 datatype but not the tensor-memory hardware, not the SMEM budget, not FlashAttention-4, and not the bandwidth headroom that makes B200 inference look easy. We brought up GB10 because it is the right surface for kernel validation, single-node smoke tests, and end-to-end architectural sanity checks under the unified-memory ceiling — and because the cheap ways it fails are the cheap ways our customers' GB10 deployments will fail. This is what it took to get a clean training step.

## 1. What GB10 actually is

The first thing to internalise is that `sm_121a` is not a small `sm_100a`. The Blackwell umbrella covers two architecturally distinct chips: datacenter (`sm_100a`, B200) and consumer-class (`sm_120a` RTX 5090, `sm_121a` GB10). NVIDIA's own forum reps put it bluntly: GB10's tensor cores are "closer to the GeForce Ampere-style MMA model". RT cores and DLSS silicon took the die budget that would have gone to TMEM and `tcgen05` on the datacenter parts.

| Property | B200 (sm_100a) | GB10 (sm_121a) |
|---|---|---|
| SM count | 132 | 48 |
| Memory | HBM3e ~8 TB/s | LPDDR5X ~273 GB/s, 128 GB unified |
| Dynamic SMEM budget | ~232 KiB | ~99 KiB |
| `tcgen05.*` family / TMEM | yes | absent |
| 2-SM TMA multicast | yes | cluster cap 1, effectively absent |
| Hopper `wgmma.mma_async` | n/a (deprecated) | n/a (deprecated) |
| FlashAttention-4 cubins | yes | rejected at driver |
| Tensor-core peak | ~2,250 BF16 / ~9,000 FP4 TFLOPS | ~100 BF16 / ~400 FP4 TFLOPS |

The folk wisdom that "FP4 doesn't work on consumer Blackwell" is wrong but understandable. FP4 does work on `sm_121a` via warp-level OMMA. What does not work is the `tcgen05`-coupled UTCOMMA path, which is what most CUTLASS NVFP4 examples hard-code. The OMMA-based examples actually run on GB10; the TMEM-coupled ones do not.

## 2. The toolchain dance
Bringing up our hybrid pretraining recipe on GB10 took five separate fixes before the first iteration produced a finite loss. None of them were exotic; all of them were undocumented in the obvious places. ### ptxas Triton ships its own `ptxas` (12.8) which does not know what `sm_121a` is and bails with `Value 'sm_121a' is not defined for option 'gpu-name'`. Point Triton at the system CUDA 13.0+ ptxas: ```python if not os.environ.get("TRITON_PTXAS_PATH"): for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]: if ptxas and os.path.exists(ptxas): os.environ["TRITON_PTXAS_PATH"] = ptxas break ``` ### `is_big_gpu` PyTorch's inductor refuses `max_autotune_gemm` if the GPU has fewer than 68 SMs. GB10 has 48. Two lines: ```python os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1" import torch import torch._inductor.utils as inductor_utils inductor_utils.is_big_gpu = lambda index=0: True ``` Some Triton configs then fail with shared-memory errors during autotune. That is fine; autotune handles it. The 99 KiB SMEM budget on `sm_121a` is lower than the SM100 default tile shapes assume, and every kernel sized for B200 will overflow until you re-tile it. ### MFU The default MFU calculation in our trainer divides by H100's BF16 peak (~989 TFLOPS), which on GB10 reports a depressing single-digit percent. With the correct denominator (~62 TFLOPS BF16, ~500 TFLOPS NVFP4), MFU comes out closer to low double digits, which is roughly what the silicon can do given the bandwidth wall. ### Liger graph break `LigerFusedLinearCrossEntropyFunction` calls `target_mask.sum().item()` internally, forcing a `torch.compile` graph break and tanking Liger's throughput below the unfused baseline.


## 3. The kernel layer: TileLang wins, cuTile is a dead end on this box
The ensemble's hot path is the Mamba-3 MIMO backward-of-backward (`bwd_bwd`) kernel. We tried three independent paths to beat the TileLang baseline on GB10. All three lost. ### cuTile Python rewrite The most thorough attempt. Five algorithmic variants — fused monolithic, nested `@ct.function` per phase, 3-kernel split, hoisted loop invariants, full `ct.static_iter` unroll — all regressed against the 2-kernel A/B split baseline. The full unroll was several times slower. The 3-kernel split that won by a third on B200 (TMEM, 228 KiB SMEM) regressed by a few percent on GB10. The launch-overhead vs live-set trade-off flips the moment you change SMEM budget. The lesson, treated as a hard rule: never assume a cuTile algorithmic variant that wins on one GPU will win on another. Re-sweep on the target hardware. ### CuTe DSL hot-path port The most fun and the most humbling. We got `cute.nvgpu.warp.MmaF16BF16Op` + TMA + persistent scheduler running on `sm_121a` out of the box (via the GeForce-Blackwell dense-GEMM pattern — pass `"sm_120"` as the SmemAllocator capacity key, do not use `CUTE_DSL_ARCH=sm_120a` overrides which the cubin loader rejects). The hand-written batched GEMM at `L=256` ran in roughly 10 microseconds. `torch.bmm` on the same shape ran in roughly the same time. cuBLAS on GB10 already matches a hand-written CuTe DSL kernel at small BF16 shapes. The TileLang advantage is not GEMM efficiency; it is that TileLang fuses about ten GEMMs plus on the order of 150 elementwise ops plus rotary plus reductions into one CUDA kernel with 16 CTAs each running 16 chunks in on-chip state. cuTile Python structurally cannot do that — it has to split into at least two kernels with gmem temps. The roughly 4x gap is the kernel-structure tax, not the instruction tax. ### Triton M²RNN autotune sweep The most anticlimactic.


## 4. NaN, NaN, NaN: a bisect that wasn't
The hardest two days of this project had nothing to do with kernels and everything to do with `grad norm: nan`. Symptom: the canonical multi-GPU H200 hybrid training was producing finite gradients on day one ("the golden run") and `grad norm: nan` on every iteration two days later. The obvious suspect was a recent commit that rewrote the MTP and main-head Liger CE patches from `reduction="none"` to `reduction="mean"` plus broadcast — explicitly to fix a silent grad corruption from an upstream Liger issue. Clean hypothesis: the "fix" broke training. Empirical reality: not the fix. We ran five mutations at HEAD — `MTP_DEPTHS=0`, `CPPMEGA_PREFER_NATIVE_HOPPER_CE=1`, vanilla CE on logits, revert mamba3 regional compile, drop selective recompute. All five produced `lm loss ~12 (finite), grad norm: nan` on iter 1. The bug was upstream of all four candidates. So we set up a proper bisect on a clean checkout, with `PYTHONPATH` precedence and a `cppmega.__file__` import pre-check. Then we tested the claimed-golden commit itself under the same environmentironment. NaN. Iter 1: finite loss, NaN grad norm. Same on the other claimed "last known finite". Same on HEAD. Three commits, including the one that allegedly produced healthy gradients twenty-four hours earlier, all NaN under the same environmentironment. Conclusion forced by the data: the regression was not in our source. The software environment had drifted between the golden measurement and our bisect. Likely candidates, in order: 1. The vendored `megatron-lm` checkout had no `.git` directory. The Megatron-LM version was unverifiable. Anyone could have overwritten it. 2. The `state-spaces-mamba` fork carried uncommitted patches (an FP32 upcast of `dd_dt + self.dt_bias` before softplus, GQA branch changes in the backward kernels).


## 5. The other NaN: a mutation we made ourselves

While that NaN was being investigated, GB10 produced a different and equally instructive failure: `cudaErrorMisalignedAddress` at `mamba_mimo_fwd_kernel`, immediately on iter 0, during forward.

Root cause traced in an afternoon, once we stopped trusting the env gate. A "P1: enable TMA + warp specialization in Mamba3 MIMO kernels" commit had at some point been applied to the installed `mamba_ssm` site-packages files, not to a copy. The the public Mamba patch application helper helper had no restore path. The env gate `CPPMEGA_MAMBA3_P1=0` correctly skipped applying the patch on the next run — but the disk state was already mutated. Every subsequent Python import picked up the patched kernels regardless of the env var.

The diff was tiny:

```
-        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
-        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
+        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
```

Plus `@autotune(...)` enabled. On the pinned TileLang revision, the TMA-lower path produces bulk-copy descriptors that assume aligned multi-byte boundaries. Combined with the tile shapes in `mamba_mimo_fwd_kernel`, that produces unaligned addresses on `sm_121a` — the exact alignment-bug class that the CUTLASS sm_120/sm_121 issue tracker catalogues.

Two lessons. First: never patch installed site-packages in place. The linear-CE patch already does this correctly via monkey-patch at import time; the mamba3 P1 patch needs to be reworked to write to a `mamba_ssm_p1/` shadow tree and patch at import. Second: env gates do not protect against irreversible disk mutations. If your "off" path leaves the system in the "on" state, your gate is a label, not a switch.

## 6. What we did finally validate on GB10

After the P1 disk state was reverted, we ran a 13-layer hybrid cut (1 MLA + 3 DSA + 4 MoE + 4 Mamba3/M2RNN + 1 MTP) end-to-end on a single GB10 and got finite gradients across every config we could reasonably build:

- BF16 with unfused attention, several iterations: finite, healthy loss decay.
- FP8 tensorwise at small and medium MBS: finite across tens of iterations.
- Plus TileLang SparseMLA BF16, Liger MTP, Liger main-head, and DSA indexer fused: finite, byte-identical iter-1 grad to the no-SparseMLA run.
- Plus the full runtime configuration (`CPPMEGA_NGRAM_HASH_ENABLED=1`, `CPPMEGA_STRUCTURE_ENABLED=1`, `MAMBA3_MIMO=1`, `MAMBA_NUM_GROUPS=8`, `MAMBA_RECOMPUTE=1`): finite, grad norm decaying smoothly across ten iterations.
- Plus the true per-layer dims (hidden=3584, ffn=18944, 28 heads) at moderate MBS: finite.
- Plus the canonical MBS with `CPPMEGA_INDEX_CACHE=1`: finite, peak memory in the high-80s of GB, validation PPL in the expected band.

That last one is as close to the canonical golden config as a single GB10 can physically run. Every component that the full training runtime uses and that fits on `sm_121a` produces finite gradients on GB10. The NaN that haunted the multi-GPU system lives in the intersection of EP=8 collective backward, megatron-lm SHA drift, and TE FP8 tensorwise behaviour, none of which a single GB10 can exercise.

## What we kept and what we threw away

Kept: TileLang for Mamba3 MIMO `bwd_bwd` on GB10, the OMMA-based NVFP4 path, RHT-disabled NVFP4 recipe, the system `ptxas` for Triton, the `is_big_gpu` patch and `capture_scalar_outputs` knob, GB10 as the validation surface for kernel correctness and single-node smoke tests, env-pinned and SHA-pinned dependencies for any "golden" claim, monkey-patch-at-import for kernel mutations.

Threw away: cuTile for the Mamba MIMO forward-backward-backward kernel path on this box, FA4 source builds (silicon-blocked), trtllm-gen FMHA on SM12x, in-place mutation of installed site-packages, "the same recipe wins on every Blackwell" as a working assumption, and any throughput claim from a system whose Megatron-LM checkout has no `.git` directory.

If we had a do-over: pin megatron-lm to a SHA in a real `.git` checkout, snapshot the mamba_ssm fork state at every "golden" measurement, and never let a patch helper write to installed site-packages. Those three rules would have saved most of the time behind this writeup.

## References

- public vendor documentation on GB10 / DGX Spark
- public CUTLASS and Triton documentation relevant to `sm_121a`
- bring-up notes and validation summaries
