---
title: "Kernels that pay for themselves"
description: "Which custom kernels and fused paths in the POC are worth their maintenance cost, which ones are borderline, and which ones belong behind a fallback or in experiments."
date: "2026-04-18"
tags: ["triton", "kernels", "performance", "moe", "mla", "mamba3"]
---

Every custom kernel is a liability until it proves otherwise. It ties you to compiler behavior, backend quirks, memory-layout assumptions, and a testing burden that plain PyTorch code does not have. The only honest reason to keep one is that it pays for itself. In the POC, a handful of kernels clearly do. Others are still good experiments, but they do not deserve unconditional residency in MegaCpp.

## Why MegaCpp cares

MegaCpp is downstream of a research repo, not a blank slate. That means the real question is not “can we write a fused kernel for this?” but “should this fused path survive contact with a long-lived product codebase?” If a kernel only wins in a narrow benchmark and loses on maintainability, it should stay in experiments. If it removes a persistent bottleneck across real training runs, it belongs in the platform.

The repo gives us the right evidence surfaces for this decision. `goodput.py` measures useful training time against wall time. `temporal_perf.py` measures throughput and memory over time. `report.py` turns that into comparable reports. `receipt_schema.py` gives ablations a stable structure. The rest of the answer comes from the actual fused files and the changelogs.

## What we built in the POC

The surviving kernel set falls into three buckets: definitely pays for itself, conditionally worth keeping, and not worth a permanent product dependency.

| Surface | Files | Why it exists | Keep level |
|---|---|---|---|
| Fused MoE path | the fused MoE kernel module | Replace a multi-stage route/permute/pad/GEMM/unpermute pipeline | Keep |
| Mamba fused update path | fused Mamba-related modules and changelog references | Remove repeated state-update overhead from a hot recurrent lane | Keep |
| Fused residual and mHC helpers | Public residual and bias-dropout-add helpers | Collapse repeated elementwise work around every block | Keep |
| Fused RoPE Q+K | Public Triton RoPE kernels | Share loads and reduce launches on a hot attention-adjacent path | Keep |
| MLA fused RoPE | Public MLA rotary kernels | Narrow MLA-specific hot-path win | Keep |
| MLA fused projection | Public MLA projection kernels | Reduce projection-stage traffic with custom autograd | Borderline |
| Custom backend dispatch wrappers | Public backend dispatch layer | Choose vendor/library fast paths with stable fallbacks | Keep |
| Small one-off fusions without repeated hot-path evidence | various experiments | Save a little work without moving step time enough | Do not keep by default |

The best example of a kernel paying for itself is the fused MoE kernel module. The file is explicit about what it replaces. The standard path has distinct routing, permutation, padding, grouped compute, unpadding, and weighted recombination stages. The fused path reduces that to a much tighter route-sort-compute-scatter flow and keeps multiple implementations available depending on backend support. That is the model of a justified complex kernel: a large avoided memory-movement bill, a fallback path for correctness, and a hot enough usage pattern that the savings recur throughout training.

The Mamba fused path belongs in the same top tier. The related posts and changelog material describe a fused trapezoidal update replacing chains of small elementwise work around state-space recurrence. That kind of kernel usually pays for itself because recurrent layers amplify launch overhead: a “small” inefficiency repeated across blocks and timesteps becomes a real budget item. MegaCpp should keep that class of optimization whenever the hybrid architecture still relies on those layers.

The public fused residual helper is a more subtle but still convincing keeper. The file contains fused residual-add, residual-scale-add, lerp-style mixing, and branch-composition helpers. The comments in the file point to profiler evidence where the unfused path consumed a meaningful fraction of the step. That is exactly the threshold we want. Residual math is boring, but it happens everywhere. Boring repeated work is often where fusion pays for itself fastest.

The public bias-dropout-add helper is also a good lesson. It is not a giant custom Triton kernel; it leans on `@torch.compile` to let bias, dropout, and residual addition become a visible optimization unit. That still counts as a fused path worth keeping. MegaCpp should prefer this kind of low-drama, compiler-visible fusion over bespoke kernels when the result is comparable.

The public Triton kernel pack contains another honest keeper: fused RoPE Q+K. Applying RoPE to Q and K in one launch with shared trig loads is the kind of narrow improvement that is easy to underestimate. But it sits on a hot path, has a clear semantic contract, and does not drag a giant maintenance surface behind it. That is a textbook “small but worth it” kernel.

There is also a useful negative lesson in the same neighborhood. Some kernels look attractive because they are mathematically neat or locally elegant, but they do not remove enough whole-path work. A path that turns three tiny elementwise ops into one kernel is not automatically a product win. If it does not sit on a repeated hot path, if `torch.compile` could have handled it anyway, or if the resulting code becomes brittle across backends, it does not pay for itself. The POC is most valuable where it already internalized that distinction.

The MLA story is split. The public MLA fused-RoPE kernel is easy to justify: it encodes MLA-specific query and key/value rotary handling in one narrow Triton path and avoids extra staging work. `fused_mla_projection.py` is harder. Its goal is good: fuse down-projection, normalization, and up-projection while recomputing intermediates on backward. But that kind of custom autograd fusion is exactly the area where vendor libraries and compiler improvements can catch up quickly. If the win shrinks, the maintenance cost dominates. So the right conclusion is not “delete it,” but “treat it as conditional.”

`kernels.py` deserves mention because a kernel can pay for itself indirectly. The dispatch layer routes cross-entropy and normalization to the best available backend, with stable fallbacks for other devices and modes. A solid dispatch surface can be more valuable than yet another custom kernel because it lets the product adopt fast vendor code without scattering backend logic through the model. That is a complexity-saving mechanism, and complexity saved is part of the payoff equation.

That is especially visible in the loss path. `kernels.py` contains several cross-entropy variants: current plain execution, Liger-routed execution, CCE-backed execution, chunked execution, and row-sharded vocab-parallel handling. The point is not that every one of those is a “kernel we keep.” The point is that the dispatch layer protects the model from having to know which exact fast path is valid on which lane. In production, this can save more engineering time than squeezing one extra micro-optimization into a custom Triton file.

Another sign that a kernel pays for itself is when the file contains explicit fallback reasoning rather than just a fast path. the fused MoE kernel module is good here. It can use Triton, grouped GEMM, or a loop-based reference implementation. `fused_bias_dropout_add.py` similarly offers fused and unfused closures. `fused_residual.py` repeatedly falls back to plain PyTorch paths when needed. This is not accidental defensive programming. It is part of the kernel’s economic case. A fused path that cannot degrade gracefully is much more expensive to carry.

## How it lands in MegaCpp

MegaCpp should keep the kernels and fused paths that meet four conditions.

1. They operate on a hot path that repeats throughout training.
2. They remove whole stages of movement or launch overhead, not just one micro-op.
3. They have a clean fallback path.
4. They can be judged with goodput and receipt data, not just microbenchmarks.

By that standard, the default keep set is straightforward: fused MoE, Mamba fused update, fused residual and mHC helpers, fused RoPE Q+K, MLA fused RoPE, and backend dispatch layers like `kernels.py`.

The conditional set is also straightforward: larger MLA projection fusion and any custom path whose advantage depends too heavily on a specific nightly or backend version.

The product rule should look something like this:

```toml
[fast_paths]
fused_moe = true
mamba_fused_update = true
fused_residuals = true
fused_rope_qk = true
mla_fused_rope = true
mla_fused_projection = "experimental"
backend_dispatch = "required"
```

Again, that is not a literal repository config. It is the right operational contract extracted from the code.

There is also a governance implication for MegaCpp. The kernel catalog should not grow by default. New fused work should have to displace an existing bottleneck or remove a class of backend pain. If a proposal only says “this benchmark kernel is faster,” that is insufficient. It should also explain why a compiler-visible fusion, a vendor path, or a dispatch-layer solution is not enough. The product should be biased toward fewer custom kernels, not more.

## Ablations and what we kept

The repo already contains the ingredients for a keep-or-drop policy that is better than taste.

First, `goodput.py` separates useful training work from badput categories such as compilation and idle time. That keeps kernel conversations honest. A path that lowers one kernel’s local runtime but increases compile churn or synchronization waste can still lose at the run level.

Second, `temporal_perf.py` records step history, tokens, commits, and peak memory. That means the team can see whether a fused path makes training more stable over time or just front-loads wins into a short benchmark.

Third, `receipt_schema.py` turns ablations into structured artifacts. If a fused path matters, it should survive comparison under a stable receipt schema. If it does not, then the team is keeping it on faith.

This matters because “kernel value” is often nonlinear. A path can look neutral in a microbenchmark and still pay for itself in a real run by stabilizing memory use, avoiding padding blowups, or preserving compile-friendly graphs. It can also look great in isolation and lose at the run level because it adds compile churn or makes fallback behavior worse. Without structured reports and receipts, teams tend to remember only the impressive local benchmark and forget the operational bill.

What this suggests for MegaCpp is simple:

- Keep fused MoE because it removes whole categories of padding and dispatch overhead.
- Keep the Mamba fused path because recurrent-state math punishes unfused execution.
- Keep fused residual and mHC helpers because they save work on every block and comments in the file already connect them to profiler pressure.
- Keep fused RoPE Q+K and MLA fused RoPE because they are narrow, understandable, and hot.
- Keep `kernels.py` style dispatch surfaces because they reduce product complexity while preserving access to fast backends.
- Treat larger MLA projection fusion and similar custom autograd stacks as experiments until they continue to beat baseline libraries in real training reports.

One more practical filter is readability of the contract. `fused_rope_qk` and MLA RoPE are easy to describe in one sentence. So are fused residual helpers and the MoE stage collapse. When a kernel cannot be explained succinctly in terms of which whole-path costs it removes, that is often a warning sign that the complexity is outrunning the payoff. MegaCpp should be skeptical of any kernel whose justification depends on a long chain of caveats.

## Production checklist

- Require every kept kernel to have an explicit fallback path.
- Judge kernels by run-level goodput, not microbenchmarks alone.
- Use `temporal_perf.py` and `report.py` for before/after comparisons.
- Require receipts that validate under `receipt_schema.py` for any keep/drop decision.
- Prefer compiler-visible fusion such as the `fused_bias_dropout_add.py` pattern when it delivers comparable wins.
- Re-review conditional kernels whenever backend libraries improve.

## References

- `kernels.py`
- `triton_kernels.py`
- the fused MoE kernel module
- `fused_residual.py`
- `fused_bias_dropout_add.py`
- `fused_mla_rope.py`
- `fused_mla_projection.py`
- `goodput.py`
- `report.py`
- `temporal_perf.py`
- `receipt_schema.py`
- `CHANGELOG.md`
- `CHANGELOG_GB10.md`
