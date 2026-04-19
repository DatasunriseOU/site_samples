---
title: "Upstream PRs we wrote for TileLang and Megatron-Core"
description: "A focused walk-through of the TileLang and Megatron-Core upstream PRs we have prepared: the bug, the fix, and what each contribution unblocks in our training stack."
date: "2026-04-18"
tags: ["upstream", "tilelang", "megatron", "kernels", "open-source"]
---

This is the focused tour of the TileLang and Megatron-Core contributions in our upstream queue. Six packs, two repos, one common shape: every one of them is a bug that crashed or silently corrupted a real training run before it became a public reproducer and write-up. Each section answers what broke, what the fix is, and what it unblocks.

## Why MegaCpp cares about this

TileLang (the kernel-DSL surface we use for every Mamba3 backward kernel and for SparseMLA) and Megatron-Core (parallelism, dispatch, and module plumbing) are two of our most aggressive dependencies. Both move quickly. TileLang in particular ships maintainer PRs faster than we can keep up; the rebase tax on local patch stack compounds, which is why upstreaming is the cheaper path in the long run.

The packs in this set break into three sub-shapes. Compiler/lowering bugs that block a path the kernel author intended to use (TileLang TMA bulk-copy on rank-3 smem; FloorMod divide-by-zero in `LayoutInference`). Dispatcher/integration gaps in Megatron that crash or silently slow training on normal hardware (Hopper FLCE `ValueError`; the Mamba `LinearCrossEntropyModule` rebase-miss; `Float16Module` blanket bf16 casting Mamba3's fp32 contract). And kernel-level memory rewrites that do not change the math but cut the working set by an order of magnitude (DSA `_compute_index_scores`).

In practice, we validate each fix against a real training workload before we turn it into a clean upstream reproducer. That keeps the public submission grounded in behavior we have already seen under production-shaped conditions.

## How we validate the work

The first version of a fix usually lands in the codebase that exposed the bug. From there we reduce it to a smaller public reproducer and a clean upstream-facing patch. The mechanics are intentionally simple: isolate the failing condition, confirm the fix on the real workload, then restate it in a form maintainers can run and review quickly.

Every pack has a self-contained reproducer. It depends only on the target library at the referenced upstream revision plus a small dependency set. It prints a clear sentinel when the bug fires and another when the fix is validated. The executable result is the source of truth; the write-up is there to explain it.

## Pack 08 - TileLang TMA bulk-copy on rank-3 smem

The bug, in one line: `LowerBulkCopy` used to assert `shared_layout->InputDim() == 2` and refused to lower any rank-3 (or higher) shared-memory descriptor, which crashed the Mamba3 MIMO backward kernels the moment we flipped TMA lowering on for Hopper.

The fix is already upstream. TileLang PR #746 (merged 2025-08-21) replaces the hard `ICHECK(InputDim()==2)` with a `LOG(WARNING)` plus a fallback to `LowerNormalCopy`. Rank-3+ smem layouts now compile; TileLang prints a warning and emits a non-bulk `cp.async` instead of aborting. Follow-ups #761 (1D TMA support) and #2005 (1D TMA regression test) cover the same area.

The pack still exists as regression coverage. Mamba3 MIMO backward kernels structurally use three rank-3 smem descriptors (`qk_dot_shared` is `[chunk_size, R, R]`; Q/K loads land in `[chunk_size, R, N]`). They rely on the warn-and-fallback to compile under `TL_DISABLE_TMA_LOWER=False`. A silent revert would crash the build with a generic TVM error deep in a kernel log instead of a clean test signal, so pack 08 now serves as a CI tripwire: three configurations (3D smem bf16, a 4D variant, and the production-shaped MIMO `qk_dot_shared` layout), each validated on H200 SXM and on GB10 (sm_121a). It buys us the ability to keep `TL_DISABLE_TMA_LOWER=False` on Mamba3 MIMO bwd, which is the precondition for the rest of the TMA pipelining work.

## Pack 13 - TileLang FloorMod divide-by-zero in `LayoutInference`

The bug. Inside a `T.Parallel(...)` loop body, indexing of the form `csr % R` / `csr // R` (with `R` a Python-int compile-time constant closed over by an outer `@tilelang.jit` function) crashes `LayoutInference` with `Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`. The crash fires in `tvm::arith::TryConstFold<tvm::tir::FloorMod>` while normalizing the iter-map for the parallel loop's output buffer fragment. The divisor has constant-folded to `0` even though the real Python value is `R = 4`; downstream substitution would resolve it, so the right behavior is to defer rather than abort.

The trigger is the kind of indexing pattern that appears after the rank-3 shared-memory flatten used in pack 07: a `T.Parallel(fused_chunk_size, N)` loop adding a per-`R` bias via `q_bias_frag[csr % R, n]`, followed by a loop that decomposes `csr` into `csr // R` and `csr % R`. With `TL_DISABLE_TMA_LOWER=True`, the kernel compiles; with it `False`, normalization fires on the transient zero and aborts. The crash is host-side, so no CUDA device is required to reproduce it.

The fix is upstream territory. The cleanest patch is in TileLang's TVM constant-folding path: have `TryConstFold<FloorMod>` return `NullOpt` when the divisor transiently folds to zero instead of asserting. The architecturally cleaner fix is in the iter-map normalization pass: preserve the symbolic `FloorMod` until the divisor is pinned to a non-zero `PrimExpr`. Either way the right authors are the TileLang/TVM maintainers; pack 13 is a bug report with a reproducer, not a PR.

The obvious algebraic rewrite (`csr - (csr // R) * R` in place of `csr % R`) does not work: `RewriteSimplifier` canonicalizes the subtraction form back to `FloorMod` before `LayoutInference` runs (verified against TileLang 0.1.8). The current temporary mitigation is to keep `TL_DISABLE_TMA_LOWER=True` and `TL_DISABLE_WARP_SPECIALIZED=True` on every affected backward kernel; this costs roughly 20% end-to-end throughput on H200 vs the TMA-on projection but keeps compilation alive. We flip both flags back once the upstream fix lands.

## Pack 10 - Megatron Hopper FLCE: land #3345 and add a non-Blackwell fallback

The bug. Megatron-Core's fused linear cross-entropy dispatcher on `dev` is Blackwell-only. `Platform.__init__` in `fused_linear_cross_entropy.py` raises `ValueError(f"Unsupported architecture: {cc[0]}")` for any device whose major capability is not 10. That is every H100/H200, every GB10, every A100, every Ada L40. The first call with `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear` on those devices crashes the forward pass.

The fix has two tiers. Tier A: land the open PR #3345 (`feat/hopper-kernels`), which adds the Hopper (cc=9) entry point with CuTe DSL WGMMA kernels and rewires the dispatcher so the cc=10 and cc=9 branches share a `gpu_entry` symbol. Tier B: add a soft fallback for every other capability; when no native kernel exists, emit a `RuntimeWarning` and fall back to the unfused vocab-parallel CE in `fused_cross_entropy.py`. Tier B is ~40 lines that wrap `fused_vocab_parallel_cross_entropy` with the same `forward`/`backward` signatures as the Blackwell/Hopper entries. The flag then degrades gracefully instead of being a landmine on every non-cc=10/9 host.

Why we shipped #3345 locally first. The fusion is the difference between fitting a long-context config on H200 and OOM-ing. The non-fused path materializes a `[s, b, V]` logits tensor and a same-shape grad_logits buffer, roughly 7 GiB of avoidable peak per microbatch at our hidden=4096, vocab=151552 shape. We cherry-picked #3345 onto our `dev` pin, ran a full training step on H200, and confirmed the Hopper kernels compile and bit-match `F.cross_entropy` to within bf16 tolerance. The pack is currently `Ready: N` in our checklist: we still owe a clean H200 receipt against an unpatched tree.

## Pack 11 - Megatron Mamba `LinearCrossEntropyModule` rebase-miss

PR #3226 wired `LinearCrossEntropyModule` into both `gpt_model.py` and `mamba_model.py` on Megatron's `dev` branch (merged 2026-02-04 01:47 UTC). PR #3207, "Reapply 'Add MTP support for hybrid models'", merged ~21 hours later but rebased from a pre-#3226 snapshot of `mamba_model.py`. The replay clobbered the Mamba side of #3226: the output layer reverted to plain `tensor_parallel.ColumnParallelLinear`, the `self.fuse_linear_cross_entropy` flag was dropped, and `post_process` was changed to `post_process or self.mtp_process` in a way that affected both the main decoder head and the MTP head.

Hybrid Mamba models on `dev` then cannot take the linear-CE fusion path even when the flag is set. GPT models keep the fusion; Mamba models silently fall through to the materialize-`[s, b, V]`-logits path - the same ~7 GiB of avoidable peak per microbatch as pack 10.

The fix is the diff PR #3226 originally landed: re-import `LinearCrossEntropyModule`, restore `self.fuse_linear_cross_entropy` in `__init__`, swap `ColumnParallelLinear(...)` back to `LinearCrossEntropyModule(...)` for `output_layer`, and route `forward()` through the fused output layer when the flag is set, mirroring `gpt_model.py`. One file, exactly the diff that was overwritten.

Our local workaround is a runtime class-swap in the public MegaCpp linear-CE shim, behind an environment toggle, installed at import time. It has been running in production with the Liger CE kernel routed via the same installer. The reason to file upstream anyway is that runtime monkey-patching is a regression magnet; restoring the diff in-tree makes the wiring visible to the GPT linear-CE functional tests and removes the runtime install.

## Pack 12 - DSA `_compute_index_scores` memory

The DSA indexer's score function uses `einsum("sbhd,skd->sbhsk", q.float(), k.float())` to build an `[sq, b, h, sk]` fp32 intermediate, ReLUs it, multiplies by per-head `weights[..., None]`, and reduces over the head axis to produce `[b, sq, sk]`. The intermediate is `sq * b * h * sk * 4` bytes; at `sq=sk=4096`, `b=8`, `h=32` that is 16 GiB of fp32 working set, allocated, consumed once, discarded.

The fix accumulates directly into the `[b, sq, sk]` output buffer one head at a time via `torch.bmm`. Materialize `k_bds = k.float().permute(1, 2, 0).contiguous()` once (`[b, d, sk]`, ~4 MiB), then loop over heads computing `logits_h = bmm(q_h, k_bds)`, applying `relu`, multiplying by per-head weights, and `add_`-ing into the fp32 accumulator. FLOP count and arithmetic intensity are identical; the per-head logits tile becomes the largest live tensor instead of the full `[sq, b, h, sk]` block. Working set drops from ~16 GiB to ~268 MiB, ~60x. Numerical drift versus the einsum is `max |a-b| / max(|a|, eps) = 1.9e-7` at production shape, far below any downstream `topk` stability threshold; gradient parity is verified via `torch.autograd.gradcheck` at small shape.

There is an open Megatron PR (#4039, "Fused Indexer Loss Kernel", updated 2026-03-27) that addresses the same memory problem with a split-K Triton kernel: ~60% memory saving with a 32% perf hit and TP support explicitly deferred. Our per-head streaming accumulator gets ~89% memory saving with no perf hit and no TP deferral. Opening a competing PR would stall both, so pack 12 ships as a comment on #4039 framed as a complementary approach for the BF16 fallback paths.

## Pack 16 - Megatron `Float16Module` silently casts Mamba3 fp32-contract params

`Float16Module.__init__` in `module.py` walks every parameter of the wrapped module and casts to bf16 (or fp16) indiscriminately. Upstream `Mamba3` deliberately keeps several parameters in fp32 because the TileLang scan kernel's dispatch signature requires fp32: `Q_BIAS`, `K_BIAS` (which is `C_bias`/`B_bias`), `D`, `dt_bias`, and on MIMO paths `mimo_x_bias`, `mimo_z_bias`, `mimo_o_bias`. `Float16Module` does not know about that contract and silently overrides it.

`Mamba3.forward` computes `DT = F.softplus(dd_dt + self.dt_bias)`. With `dt_bias` cast to bf16, `DT` is bf16 and goes into `mamba_mimo_fwd_kernel`, whose signature declares `DT: T.Tensor([B, H, S], T.float32)`. On stacks where the kernel's argument validation fires first you get a clean `RuntimeError: kernel mamba_mimo_fwd_kernel input DA_CS dtype expected float32, but got bfloat16`. On stacks where validation order is different you get silent garbage and `grad_norm=NaN` on iter 1. `D` and the rest fail the same way on varlen and MIMO paths.

An earlier per-forward pre-hook re-cast the fp32 params on every step, which `nsys` showed cost ~305 ms/iter in `.data.float()` copies. The current shim patches `Float16Module.__init__` to call the original initializer and then walk submodules of type `Mamba3`, restoring each fp32-contract parameter to fp32 exactly once. It installs alongside the MIMO `__post_init__` patch in the same file and eliminates the per-step copies.

Pack 16 is filed against `NVIDIA/Megatron-LM` as a generic `Float16Module` contract bug, not a Mamba-only workaround request. The proposal is for `Float16Module` to honor a per-module fp32 contract (an opt-out attribute the wrapped module sets) rather than silently rewriting dtypes. The reproducer is shared with pack 05 and split by stage: the `bf16` stage triggers the cast symptom (pack 16); the `gqa_unpatched`/`gqa_patched` stages trigger the Mamba3-side GQA bug (pack 05).

## How it lands in MegaCpp

Three of the six packs land as runtime patches in our tree and stay there until upstream catches up. The Mamba `LinearCrossEntropyModule` reroute (pack 11) is a runtime class-swap behind an environment toggle; the `Float16Module` Mamba3 fp32 restore (pack 16) is a one-shot init patch in the shim file; the Hopper FLCE bypass (pack 10) is a dispatcher probe that swaps in our Liger or Apple CCE entry when native fusion raises. Two more (TileLang FloorMod in pack 13 and the rank-3 smem regression guard in pack 08) cost throughput rather than correctness; we keep `TL_DISABLE_TMA_LOWER=True` on the affected backward kernels until upstream lands the FloorMod fix, then flip it. The DSA `_compute_index_scores` rewrite (pack 12) lands as a direct in-tree patch with no runtime probe.

The throughput math is unsentimental. Hopper FLCE (pack 10) plus the DSA indexer rewrite (pack 12) together unblock the long-context microbatch sizes we want; without them we OOM the H200 budget at the shapes the model actually ships at. The Mamba CE wiring (pack 11) closes the asymmetry between GPT and Mamba models on the same fusion flag. The `Float16Module` fix (pack 16) eliminates `grad_norm=NaN` on iter 1 of any Mamba3 training run using the upstream wrapper. The TileLang fixes (08, 13) are the ones we cannot land ourselves; they gate a ~20% Hopper backward win we are deliberately leaving on the table until the upstream FloorMod fix lands.

## Production checklist

- Every pack has a self-contained reproducer that runs against a named upstream SHA, with `requirements.txt` next to it, and prints a sentinel on success/failure.
- Reproducers stamp host capability and dependency versions in their first lines of output.
- Packs that overlap with an open upstream PR are filed as comments on that PR, never as competing PRs.
- Packs that have already shipped fixes upstream are repurposed as local regression tripwires, not re-filed.
- Runtime workarounds in our tree have an explicit "remove when PR #N lands" comment pointing at the upstream thread.
- Megatron packs run `tools/autoformat.sh` before any diff is attached.
- The "post it" decision is human-typed, not automated.
- Reproducers and pack bodies contain no host-identifying labels, non-public storage URIs, non-public branch codes, or employee names other than the named authors.
- The validation manifest is the source of truth for "ready", not the markdown body.
- Hopper FLCE remains feature-flagged in our tree until pack 10 has a clean H200 receipt against an unpatched Megatron `dev`.

## References

- TileLang PR #746 ([Refactor] Merge bulk copy and improve layout inference for bulk copy), PR #761 (1D TMA support), PR #2005 (1D TMA regression test).
- TileLang `LayoutInference` constant-folding and iter-map normalization passes.
- Megatron-LM PR #3345 (Hopper FLCE kernels), PR #3226 (LinearCrossEntropyModule wiring), PR #3207 (MTP replay that reverted #3226), PR #4039 (Fused Indexer Loss Kernel, split-K Triton).
- Module references: `dsa.py`, `tilelang_sparse_mla_fwd.py`, the public Mamba backward kernel sample, `mamba_model.py`, `gpt_model.py`, `linear_cross_entropy.py`, `fused_linear_cross_entropy.py`, `module.py` (Megatron `Float16Module`).
- TensorRT-LLM PR #12198 (the analogous DSA indexer fuse on the inference side).
