---
title: "Upstream PRs we wrote for Mamba-3, Sparse-MLA, Liger and DSA"
description: "A focused walk-through of the Mamba-3, Sparse-MLA, Liger-Kernel and DSA upstream PRs we have prepared: the bug, the fix, and where each one currently sits."
date: "2026-04-18"
tags: ["upstream", "mamba3", "sparse-mla", "liger", "dsa", "kernels"]
---

This is the focused tour of the Mamba-3 and Sparse-MLA side of our upstream queue, plus two adjacent packs (Liger FLCE backward and DSA CUDA-graph safety) that share the same shape. Eight packs across five repos: `state-spaces/mamba`, `tile-ai/tilelang`, `NVIDIA/TransformerEngine`, `linkedin/Liger-Kernel`, and `NVIDIA/Megatron-LM`. Each section answers what broke, what the fix is, and where the pack sits today.

## Why MegaCpp cares about this

Two architectural choices set the bar. Our hybrid presets put Mamba-3 SSM blocks in series with attention, so we live inside `state-spaces/mamba` and its TileLang backward kernels every step. We run Sparse-MLA in the absorbed configuration on Hopper because the working-set savings are decisive at long context, so we live inside the `tilelang_sparse_mla_*` kernel family on the same step.

Both families are research code drafted at one configuration and not yet generalized. The Mamba-3 MIMO backward kernel covers `G == 1` (MHA) and `G == H` (per-head) but not the GQA middle. The Sparse-MLA TileLang kernels hardcode the DeepSeek V3.2 dimensions (576 / 512) and bail out elsewhere. The Mamba-3 SISO backward computes the same `V @ dO^T` chunk-GEMM three times. The Sparse-MLA backward stores P/dP shared buffers in bf16 where they need fp32. None of these crash a regression test the maintainer wrote; all of them crash or corrupt our training run.

The DSA and Liger packs are here for the same reason. DSA's CPU-syncing validations (`torch.equal`, `tensor.any()`) crash CUDA graph capture, gating one of our highest-impact perf wins. Liger's `LigerFusedLinearCrossEntropyFunction(reduction="none")` returns a sensible-shaped forward tensor but corrupts gradients on any non-uniform `grad_output` because the kernel reads `grad_output[0]` and broadcasts it. Both bugs hide until you flip the flag you want to flip.

## How we validate the work

The first version of these fixes lands close to the workload that exposed the bug, then gets reduced into a public reproducer and an upstream-facing patch. In a few cases we also keep a temporary application-level mitigation while the upstream path is still open. Each pack has a self-contained reproducer with sentinels such as `BUG_REPRODUCED`, `FIX_VALIDATED`, and `MEMORY_SAVE_VERIFIED`; the executable outcome is the source of truth.

## Pack 02 - SparseMLA dimension generalization

The fused SparseMLA TileLang kernels are hardcoded for DeepSeek V3.2 dimensions. The dispatcher admits only `query.size(-1) == 576 and v_channels == 512`; the forward path asserts `dim_plus_tail_dim == 576`; the backward path hardcodes `D = 512`; both also assert `dim == next_power_of_2(dim)`. Any other MLA shape (for example `d_total=128`, `v_channels=64`) falls through to the unfused path that materializes the full `[B, H, S, S]` attention block and can OOM at long context.

The fix is four small edits: drop the `dsa.py` guard; plumb `d_v` through `SparseMLA.apply()` so the autograd Function knows the value channel count; relax the kernel-level assertions to `dim % 16 == 0` (the real warp-op constraint); read `D` from `o.shape[-1]` (or the new `d_v` argument) in the backward instead of hardcoding `512`. The kernel is already parameterized over dimensions; nothing about the math changes.

We shipped this locally because without it the fused path is dead code for our hybrid presets. The pack is honest about validation scope: the reproducer validates dimension plumbing (compile and launch at `d_total=128`, `v_channels=64`), not end-to-end convergence parity or fp64 gradcheck for the generalized kernel. The precision fix in pack 14 is intentionally separate so the dimension patch can land independently.

## Pack 03 - SparseMLA FP8 dispatch hazard

With FP8 training, Transformer Engine wraps tensors in `QuantizedTensor` (Float8Tensor). The wrapper lies in several ways: `.dtype` returns the logical bf16, hiding FP8 storage; `.data_ptr()` returns NULL (real data is at `._data.data_ptr()`); `.to()`, `.contiguous()`, `.reshape()` do not unwrap; only `.dequantize()`, `.float()`, `.permute()`, `.unsqueeze()` do. Handing those wrappers to the TileLang SparseMLA kernel gives the kernel NULL data pointers.

TE caveat: with Transformer Engine 2.13+ the `__torch_dispatch__` hook silently auto-dequantizes `Float8Tensor` on raw CUDA dispatch, so the `RuntimeError: data pointer expected non-NULL` no longer fires. Instead users pay silent 2x memory bandwidth - they asked for FP8 and got auto-dequantized bf16. The underlying hazards remain. The dispatch fix is about correctness of intent, not crash prevention on 2.13+.

The fix detects `QuantizedTensor` inputs in `_fused_sparse_mla_absorbed()` and dispatches to an FP8-aware variant (`SparseMLA_FP8` with `T.float8_e4m3fn` GEMMs, 2x WGMMA throughput on Hopper). For models without that variant, `.dequantize()` before the kernel call - correctness fallback that loses the FP8 win. The pack is filed against `NVIDIA/TransformerEngine` because the bug is in the wrapper's contract, not the kernel.

## Pack 14 - SparseMLA backward precision (P/dP in `accum_dtype`)

In `tilelang_sparse_mla_bwd.py` the shared-memory buffers `P_shared_cast` and `dP_shared_cast` are allocated with `dtype` (bf16) before being consumed by the dKV gradient GEMM. P and dP have a wide dynamic range (`exp` of scaled scores, then multiplied by dO accumulations); bf16 storage loses precision in the dKV path and drifts versus an fp32-reference backward. The fix is one line per buffer: allocate with `accum_dtype` (fp32) instead.

Pack 14 stays at `Ready: N`. There is no checked-in example bundle yet, so the body deliberately limits itself to the code-level change and the intended validation target (improved dKV accuracy against fp32/fp64 reference, no material Hopper regression). Filing a precision fix without a numerical receipt wastes a maintainer's time.

## Pack 04 - Mamba3 SISO backward: eliminate redundant V @ dO^T

the public Mamba SISO backward DQKV kernel sample computes `tl.dot(v_block, tl.trans(do_block))` (a CHUNK_SIZE x CHUNK_SIZE GEMM) three times in the inner chunk loop, each followed by the same causal-decay mask. All three produce identical results before diverging into dADT, dK, dQ. The fix computes `vdot_block` once, applies the mask once into `vdot_masked`, and reuses it for all three consumers. Net change: -25 lines, two redundant `tl.dot` calls and two mask applications removed per chunk.

This is honestly a code-clarity / CSE-regression-robustness PR, not a perf PR. On Triton 3.7 + H200 the compiler already CSEs the three dots automatically; measured speedup at our shapes is within timing noise (sigma ~+/-2%). The value is in making the CSE explicit so that the fusion is a property of our source rather than of whichever Triton version is installed, guarding against future MLIR pass-ordering regressions, and shrinking the `RECOMPUTE_MASK` path. On H200 all seven gradient tensors (dQ, dK, dV, dADT, dQK_dot, dD, d_issm_state) are bitwise identical between original and patched, including under varlen.

## Pack 05 - Mamba3 MIMO GQA backward (missing `1 < G < H` branch)

`mamba_mimo_bwd_combined` only handles two reduction cases: `G == 1` (MHA) and `G == H` (per-head). Intermediate grouping (for example `ngroups=8`, `nheads=128`, giving 16 heads per group) hits `else: raise ValueError("G value of {G} is not currently supported!")` or, on a slightly different code path, produces silently incorrect gradients. `G=1` and `G=H` both pass; the bug only surfaces in the GQA middle.

The fix is a third branch for `1 < G < H` where `H % G == 0`. Inside it, compute the bias gradients first (`dq_bias`, `dk_bias`) by summing over batch and seq before reducing dq/dk - the bias grads have shape `[H, R, N]` and must come from the un-reduced dq/dk. Then reshape dq/dk from `[B, S, R, H, N]` to `[B, S, R, G, hpg, N]` and sum over `dim=4`. Same fix in the public Mamba varlen backward kernel sample. Roughly 15 lines per kernel.

On 8xH200 the patched kernel runs ngroups=8, nheads=128, d_inner=8192, headdim=64 at 279 TFLOP/s steady-state for 20 iterations with no NaN. The reproducer's `gqa_unpatched` stage raises; `gqa_patched` produces finite grads bitwise-identical across reruns. Out of scope: the B/C layout `(r, g, n)` vs `(g, r, n)` latent bug only triggered at TP>1 with `ngroups>1`, and the Megatron `Float16Module` cast (pack 16) which shares this reproducer but targets a different repo.

## Pack 07 - Mamba3 MIMO bwd: 3D to 2D smem refactor for TMA compatibility

TileLang's `LowerBulkCopy` requires `shared_layout->InputDim() == 2` to emit TMA copies. Mamba3 MIMO backward kernels carry three rank-3 smem descriptors that block the TMA path: `qk_dot_shared` is `[chunk_size, R, R]`; the Q/K loads in `mamba_mimo_bwd_fwd_kernel` and `mamba_mimo_bwd_bwd_kernel` land in `[chunk_size, R, N]`; the `QK_DOT` global tensor is `[B, H, S, R, R]`; two register fragments (`qk_dot_frag`, `dgamma_diag_prereduce_frag`) are also rank-3.

The fix is a flatten that does not change the math. `[chunk_size, R, R]` becomes `[chunk_size, R * R]`; every `[c, r1, r2]` indexer becomes `[c, r1 * R + r2]`. The signature `Q: [B, S, R, G, N]` becomes `Q: [B, S * R, G, N]` with callers passing `q.view(B, S * R, G, N)` (zero-copy). Smem footprint and register pressure are unchanged. Verified on GB10 (sm_121a): all 14 gradient tensors land well under the repo's 0.10 tolerance, bit-for-bit with the pre-patch TMA-off baseline within bf16 rounding. H200 perf is a follow-up comment, not a filing precondition.

Independently of TMA the flatten has merit: simpler descriptors and forward-compatibility with `cp.async.bulk.tensor.3d` once TileLang adds rank-3 TMA. Once flattened, the kernel becomes eligible for TMA pipelining on Hopper - but the TMA-on path is gated by pack 13 (the FloorMod DBZ in `LayoutInference`). The two packs land in order: 07 first (refactor, no behavior change), 13 second (compiler fix upstream), `TL_DISABLE_TMA_LOWER=True` workaround off third.

## Pack 09 - Liger FLCE `reduction="none"` backward silently corrupts gradients

`LigerFusedLinearCrossEntropyFunction.apply(..., reduction="none")` returns a `[BT]` forward loss that looks reasonable, but the saved `grad_input` and `grad_weight` are scaled in backward by `element_mul_kernel`, which assumes `grad_output` is a scalar and reads only `grad_output[0]`. Any non-uniform per-token `grad_output` (loss-mask weighting, document-boundary masking, per-token scaling) silently produces the wrong gradient for every row except the first.

`loss.sum().backward()` gives `grad_output = [1, 1, ...]`; reading the first element returns `1.0` and the math coincidentally matches `reduction="sum"` - silent pass. `(loss * loss_mask).sum().backward()` gives `grad_output = loss_mask`; the scalar read returns `loss_mask[0]` and scales every row by that one value, with measured `max|delta grad_hidden| = 4.7e-2` versus eager PyTorch (bf16 noise floor is 5e-3). Downstream: a Megatron LM-head integration with loss-mask before reduction reports `grad_norm = NaN` on iter 1 and crashes with CUDA illegal memory access on iter 2.

The fix replaces the scalar `element_mul_kernel` with a row-broadcast kernel when `grad_output` is a tensor: `grad_input.mul_(grad_output.unsqueeze(-1).to(grad_input.dtype))` for the input side, and either redo the chunked MM in backward using saved per-chunk grad_logits or recompute them. Cost is one extra chunked matmul in backward; forward memory footprint is unchanged. The pack also asks for a docstring note that `reduction="none"` was forward-only on the previous implementation.

Upstream context: issue #968 was closed without a fix; draft PR #1126 prevents `reduction='none'` backward by raising an error (safer than corruption, but a policy choice that closes the workflow); PR #1182 is in the area but does not land the row-broadcast kernel. Our pack ships the working fix as a comment on #1126. Our temporary application-level mitigation is to call Liger with `reduction="mean"`, rescale by `n_valid`, and broadcast back. That is exact when the caller's `loss_mask` equals `(labels != ignore_index)` and a uniform-loss approximation otherwise.

## Pack 01 - DSA CUDA-graph safety

`dsa.py` contains several CPU-syncing ops that are fine in eager mode and fatal under CUDA graph capture. `torch.equal(finite, expected)`, `torch.equal(key_positions, ...)`, and `torch.equal(mask[bi], ref_mask)` validations all call `.item()` internally and force a `cudaStreamSynchronize`. `_scatter_topk_into_index_mask` uses `if torch.any(idx_chunk < 0): if valid_topk.any(): ...` - two scalar reductions with branching. All trigger `cudaErrorStreamCaptureUnsupported` under `--cuda-graph-impl transformer_engine`.

The fix gates `torch.equal()` validations on `torch.cuda.is_current_stream_capturing()` so they run in eager mode but are bypassed under capture, and rewrites the branchy scatter into a branchless clamp / scatter / fixup using `any(dim=-1)` (last-dim, not scalar; no CPU sync). Verified on 8xH200 with attention, Mamba, and MoE scopes captured: training completes cleanly and loss convergence is identical to the non-CG baseline within noise. Pack 01 ships as an issue first because the relevant code is already active upstream and the maintainer can either land it directly or invite a PR.

## How it lands in MegaCpp

The Mamba-3 packs (04, 05, 07) ship as PRs against `state-spaces/mamba` in that order, to keep reviewer fatigue down on the same maintainer team. The Sparse-MLA packs split across three repos: 02 to `tile-ai/tilelang`, 03 to `NVIDIA/TransformerEngine` (the wrapper, not the kernel), and 14 held until it has a complete example bundle. The Liger pack (09) ships as a comment on draft PR #1126. The DSA pack (01) ships as an issue against `NVIDIA/Megatron-LM`. Temporary mitigations stay documented until the corresponding upstream path lands.

## Honest about what is still in review

None of these eight packs is filed yet. The Mamba bundle (04, 05, 07) is the next wave; the SparseMLA pair (02 to TileLang, 03 to TE) is the wave after; the DSA and Liger packs (01, 09) ride in Group A and Group B. Pack 14 is held until it has an example bundle. The dishonest version of this post would claim things are "in review"; the honest version is that every pack sits in the queue until the target maintainer has bandwidth, and we hold to the wave cadence so we do not flood. The upstream PRs already in flight in these areas - TileLang PR #746, Megatron #3674, Megatron #4039 - shape what we file next; our packs slot in next to them, not over the top.

## Production checklist

- Each Mamba3 pack runs on `state-spaces/mamba` at a named SHA, with the Triton/TileLang versions pinned in the reproducer's `requirements.txt`.
- SparseMLA dimension generalization (pack 02) only claims compile/launch parity at non-DeepSeek dimensions; the body explicitly notes that end-to-end convergence parity is future work.
- SparseMLA precision (pack 14) stays at `Ready: N` until an example bundle with a checked-in gradcheck and an H200 throughput receipt exists.
- FP8 dispatch (pack 03) documents the TE 2.13 auto-dequant hedge so the maintainer sees both the crash-on-old-TE and the silent-bandwidth-loss-on-new-TE failure modes.
- Mamba3 MIMO 3D-to-2D refactor (pack 07) ships with the legality-proof gradient table; H200 perf is a follow-up comment, not a filing precondition.
- Mamba3 MIMO GQA branch (pack 05) ships with the bias-gradient fix computed before the dq/dk reduction; reviewers should look at the order of those operations specifically.
- Liger FLCE pack (09) ships as a comment on draft PR #1126; the row-broadcast fix is the working fix, the docstring change is mandatory either way.
- DSA CUDA-graph pack (01) ships as an issue first; the fix is small enough that the maintainer can take it directly.
- Reproducers and pack bodies contain no non-public infrastructure references, non-public branch labels, or employee names other than the named authors.
- Every temporary mitigation has an explicit removal condition tied to the upstream outcome.

## References

- [Mamba repository](https://github.com/state-spaces/mamba)
- [TileLang repository](https://github.com/tile-ai/tilelang)
- [Transformer Engine repository](https://github.com/NVIDIA/TransformerEngine)
- [Liger-Kernel repository](https://github.com/linkedin/Liger-Kernel)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [Mamba-2 paper](https://arxiv.org/abs/2405.21060)
