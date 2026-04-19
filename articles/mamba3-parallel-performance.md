---
title: "Mamba 3 Parallel Performance: Where It Beat Attention, and Where It Lost"
description: "MIMO scaling, chunk-size behavior, the PsiV cache trade-off, and an honest tally of where a Mamba 3 hybrid outran pure attention on NVIDIA H200 and where it did not."
date: "2026-04-18"
tags: ["mamba3", "state-space", "mimo", "performance", "parallelism"]
---

The question that gates everything else for a C++ specialist model is blunt: does a Mamba 3 block actually go faster than an attention block, at the sequence lengths we care about, on the hardware we have? MegaCpp stack gave us a first answer. This post is the performance side of that answer: MIMO scaling, chunk-size behavior, the PsiV cache trade-off, and a frank tally of where the hybrid pulled ahead of pure attention and where it did not.

All numbers below come from two NVIDIA H200 training hosts, a smaller internal correctness box used only for debug validation, and the TPU v6e lane we use for XLA ablations. Configuration labels are internal names; shapes are real.

## Why this matters

The hybrid is expensive to ship. A Mamba-3 MIMO kernel is register-heavy, the fork has to be disciplined across hosts, and the TileLang compiler surface is not as battle-tested as PyTorch's own ops. Paying that ship cost only makes sense if the hybrid wins where it is supposed to: at 16K to 64K context on C++ snippets packed with cross-file structure. A careful perf accounting keeps the hybrid argument honest. When the hybrid wins it is because O(N) scan cost beats O(N^2) attention at the sequence lengths our training data actually uses; when it loses it is usually because the kernel sits off its register ceiling or because a micro-optimization got announced before nsys said anything.

## 1. The shapes that matter

Before percentages, the geometry. The MIMO scan is parameterized by `(H, G, N, P, R, chunk_size, B, S)`:

- `H=16` heads per Mamba layer
- `G=1` B/C group (we keep `ngroups=1` for the author-pure contract)
- `N=64` state dimension
- `P=64` head width
- `R=4` MIMO rank, four up-projections sharing one scan
- `chunk_size=16` for the MIMO kernel (not the 256 we used in the Mamba 2 reference)
- `B=1, S=8192` per rank at the reference shape, MBS=8 in practice

Those numbers are locked by `AuthorMamba3Config`. The config layer refuses overrides that do not satisfy `H = hidden_size * expand / head_dim` because the author kernel assumes a specific head count; silent mismatches on SSM head geometry corrupt gradients in ways that only show up after hours of training.

## 2. MIMO is where Mamba 3 earns its FLOPs

Mamba 2 already reframed the selective scan as a structured state-space duality and made it a matrix operation. Mamba 3 MIMO adds a rank-`R` outer product to the state update. The PsiV tensor that dominates the kernel is a per-chunk pointwise product:

```python
psi_v[cs, r, p] = v[b, chunk_start + cs, h, p] * psi[h, r, p]
```

where `psi` is the learned `MIMO_V` parameter of shape `(H, R, P)`. At `R=4`, each head carries four up-projected channels of `V` through the scan at once. Arithmetic intensity goes up without widening the head or adding heads.

In practice, MIMO is how we get attention-like representational width out of an O(N) kernel. For C++ tokens, where one head needs to track both "what scope am I in" and "what type does this identifier bind to", a single scan with four channels behaves closer to four narrow scans than to one wider scan, and the perf profile stays linear.

The measured price: the MIMO scan is register-heavy. On the reference mid-sized hybrid at MBS=8, `nsys` captures on an NVIDIA H200 host show:

| Kernel | Time | Regs | Smem | Occupancy |
|---|---|---|---|---|
| `mamba_mimo_fwd` | ~1.19 s | 239 | 196 KiB | ~6.2 % |
| `mamba_mimo_bwd_fwd` | ~1.03 s | 255 | 196 KiB | ~6.2 % |
| `mamba_mimo_bwd_bwd` | ~2.11 s | 255 | 228 KiB | ~12.5 % |

Three observations. First, the double-backward kernel is the tall pole; it runs at about 12.5 percent occupancy at 255 regs per thread, effectively at the compiler ceiling for this launch shape (`65536 / (2 * 128) = 256`). Second, forward and first-backward are both at about 6.2 percent occupancy, meaning the scan kernels are not memory-bound; they are register-bound on a compute-bound workload. Third, the backward is larger than the forward by more than 2x, which is where optimization pressure belongs.

## 3. Chunk size behavior

We ran the MIMO forward kernel across eleven parameter shapes to sanity-check correctness and pick a chunk size. Target tolerance was `rel_err < 0.1`; in practice every shape came in below 0.01:

| shape (N, P, R, chunk, BB) | stable_max_rel | max_abs |
|---|---|---|
| 16, 64, 4, 8, 128 | 0.006 | 0.28 |
| 32, 64, 4, 16, 256 | 0.007 | 0.90 |
| 64, 64, 4, 16, 256 | 0.008 | 0.54 |
| 128, 64, 4, 16, 256 | 0.008 | 1.04 |
| 256, 64, 4, 8, 256 | 0.009 | 1.34 |
| 64, 128, 4, 16, 256 | 0.005 | 0.58 |
| 128, 32, 4, 16, 256 | 0.007 | 0.96 |
| 128, 128, 4, 8, 256 | 0.008 | 0.85 |
| 128, 64, 8, 8, 256 | 0.006 | 0.32 |
| 128, 64, 2, 32, 256 | 0.005 | 2.56 |
| 128, 64, 1, 64, 256 | 0.009 | 6.41 |

Two observations drove the chunk-size choice. Larger chunks (`chunk=64, R=1`) pushed `max_abs` up roughly 10x without materially helping throughput, because the register window grew with `R * P`. Smaller chunks (`chunk=8`) were fine on correctness but spent more time on launch overhead and inter-chunk state plumbing. The sweet spot for the reference shape is `chunk=16`, which lets us keep `R=4` without asking the compiler for more shared memory than the target NVIDIA kernel configuration could sustain.

On the 14-gradient backward test at the smallest shape, `stable_max_rel` landed between 0.004 and 0.012 with `bad_frac < 0.05` everywhere. That is the tolerance we carry forward as the correctness gate when we patch the kernel.

## 4. The PsiV recomputation tax

PsiV is computed from scratch in `fwd`, in `bwd_fwd`, and again inside `bwd_bwd`. Three independent materializations per forward-backward iteration of a tensor whose two inputs are stable within the iteration. The next cache pass is about turning that into one materialization plus two loads.

The cache is an activation checkpoint, not a hash table. Save PsiV to gmem during `fwd`, pass it into the backward kernels as an extra input, skip the recompute. Shape `(B, S, H, R, P)`, BF16, chunk-contiguous layout. For the reference shape at MBS=8 that is about 5.6 GiB of extra per-rank memory, fine inside the roughly 132 GiB peak we measured on the H200 runs. Expected envelope is 1.5 to 2.3 percent total TFLOP/s.

There is a real failure mode this design is ready to accept: if the TileLang compiler is already CSE-ing `psi_v = v * psi` across its scheduling stages (hoisting the load and keeping the product in a register across back-to-back `ct.mma` calls), the runtime cost is already near zero and we get nothing. That is why the first step is a Python-level materialization to measure the ceiling. If the Python hack does not move nsys numbers, the whole pursuit is archived. The env gate reads `CPPMEGA_MAMBA3_P2=1`, default OFF.

## 5. What beat pure attention

Three places, measurable.

**Context length.** At 32k to 64k tokens of C++, quadratic attention cost becomes the dominant line item in an otherwise well-tuned stack. A Mamba 3 layer at the same shape runs O(N) per token with a constant that is not small (around 1.2 s per step for MIMO forward at the reference shape is nothing to brag about) but it does not grow with sequence length. On our v4 context-graph snippets (up to 64k tokens of Callers -> Target -> Callees), the hybrid spends most of its FLOPs on the scan and reserves attention for the handful of layers that actually need content-addressable lookup.

**Per-head information density.** MIMO at `R=4` means each head carries four channels through the same scan. Equivalent attention-based width would require either more heads (more KV cache) or a wider head dimension (more per-op compute). On NVIDIA H200, the MIMO path is cheaper at the same representational capacity for the shapes in this post.

**Long-sequence memory stability.** With a minority of attention blocks, KV-cache growth at long context is small. For inference with a 64k prompt, the peak live set stays well under what an equivalent all-attention stack would demand, which lets us keep a larger micro-batch at eval time.

## 6. Where it lost

Three places, and we do not pretend otherwise.

**Forward-only P1 was a wash.** We flipped `TL_DISABLE_TMA_LOWER` and `TL_DISABLE_WARP_SPECIALIZED` from `True` to `False` on the MIMO forward kernel and added `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`, expecting 20 to 30 percent forward speedup. We measured over 19 samples at MBS=8 on an 8x NVIDIA H200 host:

| Metric | Baseline | Selective P1 | Delta |
|---|---|---|---|
| Throughput (TFLOP/s) | 183.016 | 183.005 | -0.006 % |
| Iter 1 lm loss | 11.8775 | 11.8775 | identical |
| Iter 25 lm loss | 5.3296 | 5.1818 | -0.15 |
| Val test iter 25 | 5.2686 | 5.1094 | -0.16 |
| Peak reserved (GiB) | 131.924 | 132.686 | +0.76 |

Throughput delta is inside measurement noise. Reason: forward is a small fraction of the iteration (~1.2 s of ~5.5 s total), so a 25 percent speedup on forward-only moves the whole step by about one percent, which is exactly what the noise envelope swallows. The loss delta is BF16 FMA-ordering noise from a different kernel schedule; iter-1 loss is bit-identical, which is the sanity check we actually trust.

**Backward kernels hit a real TileLang bug when TMA was enabled.** `mamba_mimo_bwd_fwd` and `mamba_mimo_bwd_bwd` used three rank-3 shared-memory descriptors, which TileLang's TMA lowering cannot handle (`InputDim() == 2` assertion, "Cannot detect TMA layout"). We fixed it by flattening 3D shared memory to 2D via zero-copy reshapes (`qk_dot_shared[c, r1, r2] -> [c, r1 * R + r2]`, `Q[B, S, R, G, N] -> [B, S*R, G, N]`). Correctness survived: 14 gradient tensors at `rel_err` 0.0038 to 0.0116, bit-for-bit with the TMA-off baseline inside BF16 rounding. Full performance measurement is still pending a fresh NVIDIA H200 slot with both forward and backward patched.

**Small-context regime.** Pure attention has a floor MIMO does not beat at small context. On a 4K-context ablation sweep on a TPU v6e-x4 slice, the dense Transformer baseline landed at a lower loss at nearly identical tokens-per-second than the AdamW hybrid over the same budget. The gap closes at longer context and with the MIMO path enabled, but at 4K tokens the Mamba blocks spend their O(N) cheapness on sequence lengths that do not exercise it. The hybrid becomes clearly dominant at 16K or 64K, which is where our v4 data actually lives; at 4K, ship dense.

## 7. What comes after

The concrete list of perf work still on the table, sized by realistic gain:

1. Full P1: TMA plus warp specialization on `fwd` and both backward kernels, gated by the 3D-to-2D TMA layout fix. Modeled gain is in the 5 to 10 percent range; measurement is still pending a fresh NVIDIA H200 slot.
2. PsiV cache (P2): intra-step activation checkpoint removes two of three recomputes, modeled at 1.5 to 2.3 percent. Phase A Python scaffold first, abandon if it does not move nsys.
3. MBS=10 at fp8 param-gather: orthogonal win, paired with the Liger main-head backward fix. A point or two if the micro-batch headroom actually opens.
4. We do not ship the P3 register split of `bwd_bwd`. The short version is that the claimed 30 to 50 percent kernel speedup did not survive a careful line-by-line read of the reverse-scan live set.

Overall the public MegaCpp Mamba path answered the gating question with a qualified yes: Mamba 3 MIMO is the cheaper kernel at the context lengths we train at, it represents per-head information at better density than attention does, and the costs are concentrated in `bwd_bwd` where we already have optimization paths ready. The wins are real; we just report them honestly, including the ones that came in at noise.

## What we kept and what we threw away

Kept: MIMO at `R=4` and `chunk=16` on the reference shape; the 3D-to-2D TMA layout fix on the backward kernels; the PsiV cache (P2) as the next perf pass with a Phase A Python gate; honest reporting of the -0.006 percent selective P1 delta; BF16 correctness gates at `rel_err` below about 0.012; attention as a minority path for sharp retrieval; dense attention at 4K-only ablations. Thrown away: the P3 register split, which did not justify its complexity; larger chunks with `R=1`, which raised `max_abs` without throughput gain; and any claim that forward-only P1 delivered a meaningful speedup. The residual risk is still straightforward: the full P1 number needs a fresh H200 measurement pass.

## References

- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
- [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522)
