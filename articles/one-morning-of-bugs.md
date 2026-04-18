---
title: "One morning of bugs"
description: "A real morning's worth of upstream-library breakage during a training wave, and the operational stance we landed on: keep a patch lane and upstream the fixes once they are ready."
date: "2026-04-18"
author: "David Gornshtein"
tags: ["upstream", "debugging", "engineering"]
---

A training wave is a good stress test of every dependency in the stack at once. On one representative morning in April 2026 our hybrid Mamba-3 plus DSA recipe broke in four separate upstream libraries before lunch, and by the afternoon we had four draft upstream fixes staged in our patch lane. This is what that morning looked like, in the order the bugs appeared, and why we now treat upstream breakage as a planning constant rather than an exception.

## Why MegaCpp cares about this

The prototype sits on top of a stack under active development: nightly PyTorch, current accelerator libraries, current Megatron-Core, current `mamba_ssm`, TileLang near the tip of development, Liger-Kernel, and Triton nightly builds. Pretending that dependency graph is stable is how you lose a training window to a regression that nobody outside your run would have noticed.

The product consequence matters even more. Production MegaCpp inherits whatever we choose to work around in research. If we only patch the symptom and never write the upstream-quality fix, we carry that patch indefinitely. Waiting passively for upstream is not viable on a fast-moving stack. The middle path is a patch lane: fix locally when needed, prepare the upstream-quality explanation quickly, and retire the local patch when upstream absorbs it.

## What we built in practice

### First kernel call of the day: DSA under CUDA graphs

The first symptom was the training launcher crashing before the first optimizer step. The recipe had enabled CUDA graphs through Transformer Engine, and the indexer inside Megatron's DSA attention module failed during graph capture with `cudaErrorStreamCaptureUnsupported`.

Reading the source showed why. The hot forward path still contained CPU-synchronizing checks: several `torch.equal(...)` assertions and a branch around sentinel handling in `_scatter_topk_into_index_mask`. Those are harmless in eager mode and illegal during capture because they pull values back through `.item()`.

The fix was mechanical once we saw it. The eager-only assertions were gated away during capture, and the sentinel handling was rewritten into a branchless clamp-scatter-fixup. The lesson was not just "CUDA graphs are picky." It was that eager-mode safety checks can silently become graph-capture bugs if nobody audits them as execution modes evolve.

### Second failure, same module, different shape

Ten minutes later, a different shape of the same run died with a different symptom: a large fp32 intermediate consumed an unreasonable amount of HBM before the DSA indexer even reduced over heads.

The culprit was an implementation that materialized the full score tensor and only then reduced it. At our shapes, that intermediate was allocated, consumed once, and discarded. The run ceiling was not determined by the full step; it was determined by one transient tensor that ate all the slack.

The fix was a straightforward reduction rewrite: stream per head, accumulate directly into the smaller destination tensor, and never materialize the full four-axis temporary. The broader lesson is that memory bugs are often hiding inside mathematically correct code that simply chose the wrong intermediate.

### Third: fused linear cross-entropy with `reduction="none"`

The hybrid head uses a fused linear cross-entropy path with `reduction="none"` so we can apply a loss mask per token before reducing. The forward looked fine. The backward produced `grad_norm = NaN` almost immediately, and the next iteration crashed with an illegal memory access.

Stepping through the implementation made the failure mode clear. The backward path multiplied stored gradients by `grad_output` using a kernel that only handled scalar-style broadcasting. That is valid when `grad_output` is truly scalar or effectively uniform. It is wrong when `grad_output` carries non-uniform per-token weighting, which is exactly what a masked loss does.

This is the kind of bug that can hide for a long time because simple tests do not exercise the real reduction shape. The fix was to route the local training path through a mathematically correct reduction strategy and write the upstream explanation around the actual masked-loss use case instead of a toy example.

### Fourth: Mamba-3 MIMO with intermediate GQA grouping

The last bug of the morning came from a new preset using an intermediate GQA grouping on the Mamba-3 MIMO path. The backward path raised a literal unsupported-value error because the implementation only handled the two extremes: fully shared grouping and per-head grouping. The middle case had simply never been wired in.

The fix was to add the missing grouped branch and verify that it converged to the existing implementations at the edge cases. That same reproducer also surfaced a separate dtype issue in the wrapper stack: a module-level mixed-precision wrapper was silently casting parameters that the kernel path expected to stay in fp32.

That pair of bugs is a good example of why narrow reproducers matter. One feature addition exposed both a missing math branch and an integration-layer dtype assumption.

By mid-morning we had multiple live upstream bugs, one corner case that crossed library boundaries, and a patch lane with several new entries.

## How it lands in MegaCpp

In production MegaCpp, each of these becomes a tracked patch-lane entry with a clear retirement condition.

The DSA CUDA-graph safety fix is carried behind feature detection so it disappears automatically once we move to an upstream version that includes the fix. The DSA memory rewrite stays local until the corresponding upstream implementation is available. The fused cross-entropy issue is covered by a thin integration wrapper that avoids the broken reduction shape. The Mamba-3 grouped-backward fix lives in a small local fork until upstream supports the middle grouping case directly. The dtype-repair logic is handled as a narrow integration shim rather than a broad global override.

The implementation details vary, but the rule does not: every local fix must say when it can be deleted.

## Ablations and what we kept

The one ablation we care about on bugs like these is simple: does the workaround change the numerics?

The DSA graph-capture fix is behavior-preserving by construction. The DSA memory rewrite changes execution order but not the intended computation. The grouped Mamba backward path is checked against the two supported edge cases. The fused cross-entropy workaround is validated against a non-fused reference on the masked-loss case it replaces.

Operationally, what we kept was a disciplined patch lane: one entry per bug, one reproducer per entry, one readable explanation, and one retirement condition. What we dropped was the instinct to "just wait" for upstream on a bug that is blocking the current run. A patch lane is cheaper than a lost training day, and a clean upstream submission is cheaper than a patch that lives forever.

The broader lesson is that actively developed systems break in small, specific ways. None of these bugs was catastrophic in isolation. Each one costs somewhere between thirty minutes and a few hours if you debug it cold, and far less if you already have a patch lane ready to absorb the fix. The savings are not only in bug-finding. They are in removing hesitation about whether to patch locally, how to document it, and how to retire it later.

The patch lane is a filter, not a dump. A local patch exists to keep today's run green. A finished upstream contribution exists to stop the patch from living forever. Between them sits the real work: a reproducer we trust, an explanation someone outside the team can review, and a check that the fix is not already in flight upstream.

## Production checklist

- Local patch first when the run is blocked, then write the upstream-quality explanation the same day.
- Every patch-lane entry has a reproducer that fails without the patch and passes with it.
- Every entry pins a retirement condition.
- Import-time patches are idempotent and gated on the upstream state they compensate for.
- Run metadata records which patch-lane revision was active so later bisects stay grounded.
- Upstream breakage is tracked as an expected operational cost, not treated as a surprise.
- Upstream submissions go out only after a deliberate quality gate.

## References

- Public upstream issues and pull requests where relevant
- Public engineering notes on DSA, fused loss paths, and Mamba-3 integration issues
- Validation evidence, summarized in publication-safe form
