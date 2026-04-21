---
title: "The Mamba 3 Kernel Journey: CUDA, Pallas, TileLang, and an Honest Look at CuTe DSL"
description: "How the Mamba 3 kernel stack shipped in MegaCpp: TileLang on H200, Pallas on TPU v6e, a CuTe DSL port that never made it, and the verdicts that came out of each attempt."
date: "2026-04-18"
tags: ["mamba3", "CUDA", "tilelang", "pallas", "cute-dsl", "kernels", "H200", "tpu"]
---

Shipping a hybrid Mamba 3 plus Transformer backbone for a C++ codegen model forces the same conversation three times, once per backend: CUDA on H200, Pallas on TPU v6e, and the DSL layer on top of each. This post is that conversation written down. Short version: TileLang is how the MIMO kernels ship on H200 today, Pallas is how the SSD scan runs on TPU v6e, a CuTe DSL port of the MIMO kernel exists on disk and has not proven out, and our one serious TileLang versus CuTe comparison killed the CuTe path on ROI.

## Why this matters

DSL choice for hybrid-model kernels looks like a performance decision and turns out to be an operational one. Both TileLang and a CuTe DSL port compile to nearly identical PTX on the shapes we run; what differs is how expensive each small correctness or perf change is to land. For a training loop where patches are applied in place via an env-gated applier and reconciled across hosts by md5, the iteration cost of the kernel toolchain is the budget that matters. The kernel journey below is not "which DSL is fastest" but "which DSL lets us fix a TMA lowering bug the same day versus next week", and that question has an answer.

## 1. What we actually ship on H200

The production kernel stack for the Mamba 3 half of the hybrid is upstream `mamba_ssm` on commit `31f3d7baba`, with local working-tree patches on three TileLang files (`mamba_ssm/modules/mamba3.py`, the public Mamba MIMO backward kernel sample, the public varlen Mamba backward kernel sample) plus one correctness patch on the public Mamba SISO combined kernel sample that caches `ctx.saved_tensors` for gradient-checkpointing compatibility.

That fork is small on purpose. Patches go through an idempotent in-place applier (the public Mamba patch application helper) that crashes loudly if the upstream decorator block moves, and we md5-reconcile the working tree across both H200 hosts before any long run. The the public Mamba SISO combined sample tweak exists because accessing `ctx.saved_tensors` twice on a recomputed node raises under gradient checkpointing.

The CUDA path for Mamba 3 lives inside TileLang, not hand-written CUDA C++. The kernels that matter are `mamba_mimo_fwd`, `mamba_mimo_bwd_fwd`, and `mamba_mimo_bwd_bwd` from upstream's `tilelang/mamba3/`, plus varlen variants. Everything else (`in_proj` fusion, RMSNorm, residual) runs through Transformer Engine and regular PyTorch.

## 2. TileLang P1: what it buys, what it costs

P1 is our internal label for a targeted perf pass on the MIMO kernels: flip the disabled-by-default TMA and warp-specialization flags to enabled. Upstream ships them off:

```python
tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
```

Flipping them lets the TileLang compiler emit Hopper-class TMA descriptors for bulk gmem-to-smem async copies, and warp-group pipelining instead of plain `mma.sync`. We also add `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True` on every kernel that did not already have it, so downstream smem pressure does not push us past the dynamic cap on smaller GPUs.

This is eight `TL_DISABLE_*` flips and five new aggressive-merge insertions across four files. The patch is idempotent and env-gated behind `CPPMEGA_MAMBA3_P1=1`, default OFF. On our small internal SM_121 correctness box (99 KiB smem per SM) all three kernel groups compile cleanly with TMA and warp-spec enabled, and all eleven forward parametrized shapes pass the standard correctness tolerance. The combined backward passes at `rel_err` 0.004 to 0.012, well inside our 0.05 gate.

### 2.1 The TMA layout fix

Then we tried to run it on H200 with backward enabled. Forward compiled fine. `mamba_mimo_bwd_fwd` and `mamba_mimo_bwd_bwd` blew up inside TileLang's TMA lowering pass:

```
tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2)
is false: Cannot detect TMA layout.
```

Root cause: the backward kernels use three rank-3 shared-memory descriptors (`qk_dot_shared[c, r1, r2]` and `(B, S, R, G, N)` reads of Q); TileLang's TMA lowering only handles 2D layouts. Fix is mechanical: flatten to 2D via zero-copy reshape (`qk_dot_shared[c, r1, r2] -> [c, r1 * R + r2]`, `Q[B, S, R, G, N] -> [B, S*R, G, N]`). No arithmetic changes, just a view.

We shipped it on branch `tma-layout-fix-3d-to-2d`, with an applier and a unified diff. Correctness on the small SM_121 box survived: 14 gradient tensors at `rel_err` 0.0038 to 0.0116, bit-for-bit with the TMA-off baseline inside BF16 rounding.

On H200 we then hedged and tried shipping only the forward flip, with backward kernels left unpatched. Over a 25-iter bench at MBS=8 on the reference shape, the measured delta was -0.006 percent TFLOP/s. Inside noise. Honest read: `mamba_mimo_fwd` at ~1.2 s is a small fraction of the ~5.5 s iteration, and a 20 to 30 percent speedup on forward moves the step by roughly one percent, which our variance swallows. The full P1 win needs both the TMA layout fix and the backward flips together, and that measurement is still pending an H200 slot.

## 3. PsiV and the register ceiling

While P1 was sitting on the H200 queue, we wrote up two follow-on designs. One shipped as scaffolding; one is shelved.

The PsiV cache (P2 in our internal sequence) is the straightforward one. `PsiV` appears five times in the MIMO kernel loop body across `fwd`, `bwd_fwd`, and `bwd_bwd`, recomputed from scratch each time even though its two inputs (`psi` as a module parameter, `V` as a per-step activation) are stable within a single forward-backward iteration. Plan: save PsiV to gmem inside `fwd`, pass it to the backward kernels via `ctx.save_for_backward`, skip two of the three recomputes. Shape `(B, S, H, R, P)`, BF16, about 5.6 GiB extra per rank at the reference shape and MBS=8, fine inside our ~132 GiB peak. Modeled envelope is 1.5 to 2.3 percent total TFLOP/s.

The failure mode to check first: TileLang's scheduler may already CSE `psi_v = v * psi` across the kernel's stages, keeping the product in a register across back-to-back `ct.mma` calls. If so, the cache saves nothing. Step one is a Python-level materialization purely to measure the ceiling; if the Python hack does not move nsys numbers, the pursuit is archived. The env gate is `CPPMEGA_MAMBA3_P2=1`, default OFF.

### 3.1 Why P3 did not ship

The P3 design proposed splitting `bwd_bwd` into two kernels connected by a gmem tensor, on the theory that dropping PsiV and `qk_dot` fragments from the inner live set would let each pass fit in around 130 registers and roughly double occupancy. Pitch on paper: 30 to 50 percent kernel speedup, about 1 percent total TFLOP/s.

A line-by-line read collapsed the claim. The loop-carried `dstates_frag` is updated via `T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag, clear_accum=False)` and carried into the next reverse iteration, so pass 1 still has to hold `q_shared`, `dPhiO_shared`, and `dstates_frag` live — the exact fragments the split was supposed to drop. Separating them would cost an extra `[B, H, nchunks, chunk_size * R, P]` buffer several times bigger than `dstates_per_chunk`. Two blockers compounded the ROI problem: the small SM_121 internal box is not a viable correctness platform (upstream baseline forward fails to compile on its 99 KiB smem) and H200 access was broken on audit day. Decision: do not ship P3; pursue the PsiV cache instead, for two to three days of implementation rather than eight to twelve, at the same 1 to 2 percent envelope.

## 4. Pallas on TPU v6e

The TPU side of the same stack runs under XLA, on v6e-4 for 4K-context rapid ablations and v6e-8 for the 16K and 64K context phases. There is no TileLang path on TPU; the kernels are a mix of `torch_xla.experimental.scan` for the SSD recurrence and a chunked matmul-based reference for within-chunk mixing.

The chunked reference is where we caught most of our correctness bugs before the CUDA kernel did. The SSD dual decomposition must include both components: cross-chunk state accumulation (sequential over `nchunks = seq / chunk_size`) and within-chunk attention-like mixing. Drop the within-chunk term and tokens can only see information compressed into a chunk-boundary state; local context mixing is gone. Every reviewer proposed dropping it at some point, and the chunked reference test caught each attempt.

We use `F.rms_norm` (parameterless) for B/C QK-norm to match MegaCpp's attention QK-norm rather than `nn.RMSNorm`. Mixing the two produces orphan parameters that silently stop training. Complex RoPE uses per-dimension frequencies, not a single scalar angle per position, because single-frequency collapses the rotation to one dimension.

### 4.1 The sparse-attention Pallas kernel

A Pallas kernel does live in the TPU tree, but for a different purpose. We maintain a content-dependent sparse attention research-stack: importance scoring, query-tile union selection, and a Pallas sparse attention kernel with online softmax, parameters aligned to the v6e MXU (`Bq=256`, `l'=256`, `H=128`, `Bk=1024`). It is for the attention minority of the hybrid, not the SSM majority. Supports up to 128k sequence length with a top-`n=8` selection. Prototype committed; hardware validation receipt still open.

Main Pallas trade-off: compile time. `torch_xla.experimental.scan` rewrites the SSM loop to avoid `@while_loop` overhead. The fused HLO is cheaper per step but much more expensive to compile the first time. We eat one long compile on rank-0 at process start; without it, the Python-level loop over chunks dominates at 4K context on v6e-8.

## 5. TileLang versus CuTe DSL

The one honest A/B on the kernel stack itself was TileLang versus a CuTe DSL port of the MIMO kernel, living alongside the TileLang tree. The intent was a non-TileLang MIMO path with CUTLASS-style templates instead of a DSL compiler, so targeted changes (for example adding PsiV as an extra `fwd` output) would not fight a scheduler. After a few weeks on and off: not worth shipping, for three reasons, none about correctness.

1. **Compile-time cost.** The MIMO kernel is parameterized across `(N, P, R, chunk_size, BB)` and instantiating it through CUTLASS templates took minutes per cold build. TileLang's JIT takes seconds per shape because it caches at the TIR level. In a nightly sweep with around 20 configurations the tax is real.
2. **Targeted changes are expressible in TileLang.** The changes we actually wanted (PsiV cache, 3D-to-2D smem flatten, TMA flag flips) are all expressible inside TileLang's existing decorator block. The P1 patches are literally two lines per kernel. CuTe's advantage of "do whatever you want" mattered less than TileLang's advantage of "land a fix in an hour": when the TMA layout bug hit, the TileLang fix landed the same day, and a CuTe port would have required reflowing the smem allocation by hand.
3. **Correctness beyond `R=4` or `P=128`.** TileLang passes all eleven parametrized shapes; our CuTe port passed five. Chasing the last six was a kernel-author week the ROI math did not justify, given the low-hanging wins all live on the TileLang path anyway.

The port stays on disk as a reference implementation: the `psi_v = v * psi` identity for the P2 cache is trivially checkable side by side against it. Killing a candidate path while keeping its reference value is a reasonable end state.

### DSL comparison at a glance

| Axis | TileLang | CuTe DSL port |
|---|---|---|
| Cold compile per shape | seconds (TIR cache) | minutes (CUTLASS templates) |
| Shape coverage on MIMO | 11/11 | 5/11 |
| P1 patch size | ~2 lines per kernel | full smem reflow |
| TMA layout fix turnaround | same day | days |
| PTX parity on supported shapes | yes | yes |

## 6. What we learned about DSL choice

DSL choice is an operational cost, not a performance cost. Both TileLang and CuTe land the same PTX for the shapes we run. The difference is iteration speed: flag flips, shape tuning, and small correctness fixes land cheaper in TileLang because the compiler's surface area is closer to what we want to change. For a training loop patched in place via an env-gated applier, TileLang pays for itself on maintenance alone.

Kernel perf work is measurement-bound, not design-bound. P1 selective-forward was a wash (-0.006 percent) because forward is a small fraction of the iteration. P3 looked like 30 to 50 percent on paper; a line-by-line read collapsed that estimate to 1 to 2 percent. Real kernel wins come from an nsys capture that names the actual bottleneck, not a design that names a plausible one.

Current state on H200: upstream TileLang MIMO kernels, three working-tree patches, env-gated P1 pipeline, P2 PsiV cache scaffolded and waiting for a Phase A nsys number, P3 rejected. On TPU v6e: `torch_xla.experimental.scan` for the SSM, Pallas sparse attention path for the attention minority, chunked matmul reference as the correctness ground truth. Nothing glamorous, which is exactly the sign the kernel stack is converging.

## What we kept and what we threw away

Kept: TileLang on H200 with three upstream patches, the TMA 3D-to-2D backward layout fix, env-gated idempotent patch application, P1 behind `CPPMEGA_MAMBA3_P1=1`, P2 PsiV cache scaffolded behind `CPPMEGA_MAMBA3_P2=1`, `torch_xla.experimental.scan` on TPU v6e, and the Pallas sparse-attention research-stack for the attention minority. Thrown away: the CuTe DSL port as a production path (kept for reference), the P3 register split (rejected on ROI and platform blockers), the SM_121 box for H200-shape correctness diffs, and forward-only P1 as a default. Kernel time we do not have is still budgeted; the way to save it is to read the live set before writing the patch.

## References

- [Mamba3 trapezoid porting notes](../docs/mamba3-trapezoid-porting.md)
- [Hybrid layout notes](../docs/hybrid-layout-notes.md)
- [Mamba scan compile wrapper sample](../examples/distributed/mamba_scan_compile_wrapper.py)
- [Mamba3 PSIV cache scaffold](../examples/megacpp/mamba3_psiv_cache_scaffold.py)
- [Mamba3 3D-to-2D SMEM sample](../examples/megacpp/mamba3_mimo_3d_to_2d_smem_sample.py)
- [Mamba3 3D-to-2D SMEM nearcopy](../examples/megacpp/mamba3_mimo_3d_to_2d_smem_nearcopy.py)
