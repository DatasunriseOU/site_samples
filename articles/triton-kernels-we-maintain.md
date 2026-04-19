---
title: "The Triton Kernels We Actually Maintain In-Tree"
description: "Which custom Triton kernels we keep in the training stack, how we autotune them without getting burned, and the numerical tests that keep us honest."
date: "2026-04-18"
tags: ["triton", "kernels", "gpu", "MegaCpp training stack"]
---

This codebase has gone through several waves of "let's just write a Triton kernel for that." Most of those kernels are gone. A small set stayed, because they either saved measurable wall-clock on real training runs or removed a graph break that `torch.compile` could not otherwise close. This post is the honest list of what is currently in-tree, why each one earns its keep, and how we keep the numerics from silently drifting underneath us. The references throughout are to public modules and notes tied to the same code paths.

## Why a small set, not a large one

Custom Triton kernels are one of the cheapest-looking and most expensive-to-maintain things you can put in a training repo. They are cheap to write because the language is pleasant; they are expensive because the autotuner, the compiler and the runtime stack change underneath you across PyTorch and Triton releases, and a kernel that was a percent faster than the framework alternative six months ago can be a percent slower today, or worse, silently incorrect. Every Triton kernel in the tree is a standing dependency on three moving targets: PyTorch, Triton itself, and the Inductor lowering path that decides when to take it.

The other reason it matters is correctness drift. Tolerances written in haste become "what the kernel produces today" rather than "what the math says". We have been burned more than once. So we want a small, defensible set of kernels, each justified by a real bottleneck and each guarded by tests that compare against a recomputed reference, not against last week's output.

## What's actually in the tree

The kernels that survived live in a handful of modules: a fused RoPE family for Q and K (the oldest survivor) plus a pair of 3D row-gather kernels used by doc-masking and mixture-of-depth gather/scatter; a fused residual family covering residual-plus-scale add, residual-plus-add-plus-RMSNorm, and a few hierarchy-compose variants; the mHC dynamic-weights family with a Sinkhorn-normalized autotuned variant; the MLA-specific partial RoPE for Q and K with THD layout; a public Mamba fused trapezoidal sample that carries the trapezoidal pre-processing and diagonal-correction kernels which replaced the elementwise storm dominating Nsight Systems; a fused MoE collection that sits behind the MoE path when the cuDNN or MoEBlaze route is unavailable; and a fused ReLU-squared pair (forward and backward) for MoE-on-FFN lanes that use ReLU squared instead of SwiGLU.

We intentionally do not carry a hand-written fused RMSNorm or fused-linear-cross-entropy kernel. The public kernel routing sends `rms_norm` and `fused_linear_cross_entropy` to Liger when available, to a chunked CCE variant for the vocab-parallel shard case, and to a plain PyTorch path otherwise. Writing a custom fused RMSNorm was never worth the maintenance burden when Liger exists and gets fuzzed by a much larger user base.

| Kernel | In tree | Reason |
|---|---|---|
| Fused Q+K RoPE | Yes | Shared cos/sin, single launch, hot path |
| 3D row gather | Yes | Doc-masking/MoD gather without graph break |
| Fused residual / RMSNorm hybrids | Yes | Profile-driven on backward |
| mHC dynamic weights | Yes | Sinkhorn fusion, autotuned |
| MLA partial RoPE | Yes | THD layout, no fallback equivalent |
| Mamba3 trapezoidal pre/diag | Yes | Replaced majority of backward elementwise |
| Fused ReLU squared | Yes | Required for ReLU-squared MoE FFN lanes |
| Custom fused RMSNorm | No | Delegated to Liger |
| Fused linear cross-entropy | No | Delegated to Liger / chunked CCE |
| Bias+dropout+add Triton | No | JIT-script path is good enough |
| Custom MLA projection | No | cuBLAS grouped GEMMs caught up |
| Homegrown Mamba3 SSM scan | No | Official Mamba3 SISO kernel wins |

## The selection rule

Three criteria must clear before a Triton kernel goes into the tree. If a proposed kernel does not clear all three, it stays in an experimental lane or we delete it.

First, it must remove a real bottleneck that profiling agrees on. "Elementwise add accounts for the majority of the backward step in nsys" was the wedge that let the Mamba3 trapezoidal kernels in. A kernel that saves half a microsecond on a path that already lives inside a fused graph does not qualify.

Second, it must not prevent `torch.compile` from doing useful work elsewhere. That means the kernel has to be wrapped as a `torch.autograd.Function` with strides we trust, and it needs a pure-PyTorch fallback we can force via an environment flag. Every surviving kernel has a plain fallback path we can A/B against.

Third, it must have numerical tests at training precision and fp32, with tolerances that match the math, not tolerances that match the current output. The fused Q-K RoPE kernel is the canonical survivor by all three criteria: it shares `cos`/`sin` loads across heads and fuses Q and K into a single launch, it coexists with FA3/FA4/Pallas attention without interfering with their compile paths, and it has a dedicated parity suite covering forward, backward, GQA, fp16/bf16/fp32, and the older-SM PTX-codegen escape hatch.

## Autotune discipline

Triton's autotuner is a great tool that will happily wreck you if you treat it as free. Practical rules from repeated tuning work:

- Keep autotune configs short and meaningful. The dynamic-weights kernel carries a small, hand-curated set of `(BLOCK_M, BLOCK_N, num_warps, num_stages)` points; the Cartesian product would be much larger and most of it is identically slow.
- Pin keys explicitly. `triton.autotune(key=[...])` controls when the cache is invalidated; including only the dimensions that actually change shape behaviour avoids spurious re-autotunes when an unrelated argument flips.
- Always run autotune in a subprocess on the H200 lane. We hit a Triton stream regression where autotune crashes propagated into the live training process; the subprocess pattern turned that from "training dies" into "first compile is slower."
- Cap the workspace. Keep Inductor's GEMM autotune workspace capped by default because the workspace can OOM on the largest matmuls and poison the cache with `inf ms` picks.

The autotune cache also has to live somewhere stable. We pin it to the per-run scratch directory under the persistent volume; the boot disk is too small and shared autotune cache across runs has bitten us when an adjacent run picked a config that did not fit the current shapes.

## Numerical tests that keep us honest

Every surviving kernel has a public kernel regression test that does the same three things in some order: build a known input distribution, run the Triton kernel and a recomputed pure-PyTorch reference at fp32, and assert max-absolute and max-relative error against tolerances written next to the math, not next to the current output.

```python
def test_fused_qk_rope_matches_reference():
    q, k, cos, sin = make_qk(...)
    out_q_t, out_k_t = fused_qk_rope(q, k, cos, sin)
    out_q_r, out_k_r = qk_rope_reference(q.float(), k.float(), cos, sin)
    torch.testing.assert_close(out_q_t.float(), out_q_r, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_k_t.float(), out_k_r, atol=1e-3, rtol=1e-3)
```

The pattern is dull on purpose. The discipline is that any kernel whose tolerances change is presumed wrong until shown otherwise; bumping `atol` to make a failing test pass requires an explicit rationale in the commit record. We have caught at least three regressions where a kernel author "fixed" a failing test by loosening the tolerance and the underlying drift was real.

The other test class is graph-shape stability. For each surviving kernel we have a "compile this small model with and without the kernel and assert the FX graph is structurally equivalent" test. That catches the case where a kernel breaks a Dynamo fusion window without changing any per-element output.

## What we kept and threw away

We kept the seven kernels above, the three-criterion selection rule, the autotune-in-subprocess default, the per-run scratch for autotune cache, and the rule that any kernel without an env-var fallback gets deleted.

We threw away every "let's fuse this micro-op for fun" kernel, the global Triton autotune cache shared across runs, the practice of writing tolerances by running the kernel and copying the number, and the homegrown Mamba3 scan once the official SISO kernel landed. We also threw away a fused MLA projection that briefly outperformed cuBLAS grouped GEMMs and stopped doing so within two PyTorch nightlies; the maintenance bill outran the win.

The throughline: a Triton kernel earns its place by removing a profile-confirmed bottleneck that no upstream library will close, with a fallback we can force on, and with numerical tests that compare against the math. Anything else is technical debt waiting to be discovered by a stranger at 2 a.m.

## How a kernel actually lands in the tree

The lifecycle of a kernel is short on paper and long in practice. It starts as an experimental script run from a notebook against a pinned tensor shape and a pure-PyTorch reference. Once the script reproduces a measurable speedup against the reference on the actual training shape (not a microbenchmark shape), the author attaches the motivating Nsight Systems slice. The kernel then moves into a candidate module with the three guards: `torch.autograd.Function` wrapper, environment-flag fallback, and fp32 parity tests. Only after a real training execution record shows steady-state benefit should the kernel be wired behind the default code path. Anything that fails one of those steps stays experimental until it does not.

The reverse path is also documented. Any kernel whose parity tests degrade across two consecutive PyTorch nightlies, or whose Inductor lowering shifts in a way that makes the compiled fallback faster, gets removed. We have done this twice in the last year: once for a fused MLA projection that cuBLAS grouped GEMMs caught up with after a Hopper algorithm cache update, and once for a custom fused bias+dropout+add that was beaten by Inductor's own lowering after a 2.11 nightly. Removing kernels is part of the discipline; carrying dead code is worse than not having had it in the first place.

## The Inductor interaction

Triton kernels do not exist in isolation; they live inside graphs that `torch.compile` is also trying to optimise. Two interactions matter. The first is fusion windows: a custom kernel inserted as a `torch.autograd.Function` is a graph break, and the surrounding ops must form fusable subgraphs on either side. We have hit cases where adding a custom kernel saved 5% on the kernel itself and lost 8% on the surrounding ops because the fusion window collapsed; the kernel was reverted. The second is autotune cache invalidation: a custom kernel's autotune cache is hashed against the kernel's source plus the input shape signature, and any change to either invalidates the cache. We pin the source via the file path and document any signature change in the commit, because an unannounced signature change produces a cold-cache compile that gets blamed on PyTorch.

The practical implication is that "add a Triton kernel here" is not a local decision. It changes how the surrounding region compiles. We require a graph-shape stability test for every new kernel, comparing the FX graph of a small model with and without the kernel; that test catches the fusion-window collapse class of regression before the kernel reaches main.

## Where we still might add a kernel

There are two open candidates we have looked at and not landed. The first is a fused MoE expert sort kernel that would replace the public dispatcher's argsort-heavy path. Profiling shows argsort at a few percent of step time on the deep MoE preset, but the upstream Transformer Engine sort path has caught up enough that the win is small and the maintenance bill would be real. The second is a fused per-document RoPE reset kernel that would replace the current prefix-reset pattern. Profiling shows that pattern at sub-percent of step time, well below the wedge we require for adoption. Both stay in the experimental lane until either profiling shifts or upstream stops covering the case.

The list above is the current state. It will move; the rules will not. Profile-confirmed bottleneck, env-var fallback, parity tests at fp32, graph-shape stability test, removal when the upstream catches up. Anything that respects those rules is welcome; anything that does not is dead code from the moment it lands.

## References

- Public Triton kernel pack and fused residual helpers in the MegaCpp repo
- Public mHC, MLA rotary, Mamba3 trapezoidal, MoE, and fused-ReLU helper modules in MegaCpp
- Public backend-dispatch and change-note documentation in the MegaCpp repo
