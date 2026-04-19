---
title: "Liger FLCE reduction=none"
description: "Why Liger fused linear cross entropy can go wrong on the `reduction='none'` backward path, why `mean` stays correct, and how the scaled-mean workaround restores the intended sum contract."
date: "2026-04-19"
tags: ["liger", "flce", "cross-entropy", "hopper"]
---

This bug is useful publicly because it is not a vague optimization problem. It
is a contract bug with a narrow surface.

The broken lane is `reduction='none'` on the fused linear-cross-entropy
backward path. The known-good lane is `reduction='mean'`. The practical
workaround is to keep the kernel on the mean path and scale by the number of
valid targets to recover per-token sum semantics.

That matters because the failure mode is not only a small numerical drift. In
the real training path it can surface as corrupted gradients or NaN grad norms.

## Why the workaround is worth documenting

The workaround is not pretending the bug is gone. It is documenting a safe lane
that preserves the intended training semantics while the broken reduction path
is still unstable.

That kind of public example is worth keeping. It tells the reader which fused
path is actually safe today, not which one would be ideal if every reduction
mode were equally correct.

## Example -> article -> upstream docs

- example: [`liger_flce_reduction_none_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/liger_flce_reduction_none_nearcopy.py)
- article: [`liger-flce-reduction-none`](https://megacpp.com/blog/liger-flce-reduction-none/)
- upstream docs: Liger-Kernel issue and PR trail around `reduction='none'` plus PyTorch cross-entropy semantics

## References

- [Liger FLCE reduction none near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/liger_flce_reduction_none_nearcopy.py)
- [PyTorch cross entropy docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [Liger-Kernel repository](https://github.com/linkedin/Liger-Kernel)
- [Liger-Kernel issue #968](https://github.com/linkedin/Liger-Kernel/issues/968)
- [Liger-Kernel issue #872](https://github.com/linkedin/Liger-Kernel/issues/872)
- [Liger-Kernel PR #1126](https://github.com/linkedin/Liger-Kernel/pull/1126)
- [Liger-Kernel PR #1182](https://github.com/linkedin/Liger-Kernel/pull/1182)
