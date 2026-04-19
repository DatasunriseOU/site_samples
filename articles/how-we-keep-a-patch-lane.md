---
title: "How we keep a patch lane"
description: "The operational mechanics of running a hybrid Mamba-3 plus DSA recipe against a fast-moving stack: pinned environments, a small patch inventory, and a regular merge-back cadence."
date: "2026-04-18"
author: "David Gornshtein"
tags: ["upstream", "infrastructure", "engineering"]
---

Every serious training stack we run against nightly software needs a patch lane: a small set of local fixes, pinned environments, and a maintained inventory of what we carry and why. This post is the operational view: how we keep that lane honest while upstream code changes every week, and how we decide when a local diff can retire. The companion posts "One morning of bugs" and "External library glitches we fixed" cover the incidents themselves; this one covers the mechanism.

## Why MegaCpp cares about this

MegaCpp depends on parts of the PyTorch ecosystem that move quickly: nightly PyTorch builds, current accelerator libraries, rapidly changing model code, and a TPU lane with its own version constraints. None of those surfaces stays stable for long. If we waited for every regression to be fixed upstream before training, we would stop shipping useful work. The patch lane is how we keep moving without letting temporary fixes harden into permanent drift.

The two rules that matter most are simple: every workaround must be reviewable, and every workaround must have a retirement condition. Drift accumulates quietly when either rule slips.

## What we built in practice

### Pinned environments are the foundation

The first layer of the patch lane is not code, it is provenance. We carry several environment bundles across different hardware lanes, and each one is pinned tightly enough that we can say exactly which upstream snapshot a fix was validated against.

Nothing here is "just install the latest wheel and hope." We install in a controlled order, avoid accidental dependency resolution, and log the active stack line for every meaningful run. That makes later bisects tractable. A patch note without a known environment is not much use.

### Local forks and overlays stay small on purpose

The second layer is a small, deliberate set of local forks or overlays. The rule is straightforward: only carry a fork when there is a real diff to justify it, and every carried change needs a retirement condition.

In practice that means a small `mamba_ssm` fork for a few targeted backward-path fixes, a TileLang working tree for a handful of precision and dispatch issues, and a lighter Megatron overlay for surgical fixes that do not justify maintaining a large long-lived branch. We avoid carrying heavyweight forks of foundational libraries unless there is no cheaper path.

The important discipline is not the exact mechanism. It is that the local fix is narrow, understandable, and easy to remove once upstream catches up.

### The patch inventory is the catalogue

The third layer is the checked-in catalogue. We keep one entry per issue, together with a reproducer, a short explanation, and the current upstream status.

```text
patch inventory:
  issue notes
  status tracker
  focused reproducers
  validation metadata
```

Each entry is written in the shape of an upstream contribution: target project, problem, solution, changed surfaces, and testing evidence. Reproducers are kept small enough that an outside reviewer can run them against the pinned environment. Validation metadata records which entries were exercised end to end and which are still in preparation.

We do not use this catalogue as a diary. An entry is there because it can plausibly become an upstream contribution. If a workaround is too ad hoc to explain cleanly, that is usually a sign that the workaround itself is weak.

### Regular upstream diffing keeps the lane honest

The fourth layer is a boring but necessary check: a regular diff against current upstream. It asks, for every local fork or patch, "what are we still carrying that upstream does not?" Humans review the result. That is how a patch lane stops turning into folklore.

That check has exactly three acceptable states:

1. Zero diff: upstream absorbed the change, so we can retire it locally.
2. Non-zero diff with a matching entry: expected and tracked.
3. Non-zero diff with no matching entry: process bug, stop and explain.

That third state catches the most expensive kind of drift: a hot fix that kept a run alive but never made it into the inventory.

### Submission waves beat drive-by PRs

The fifth layer is social, not technical: we batch submissions. The early mistake was filing upstream PRs opportunistically whenever a local fix landed. Quality varied, context switching was expensive, and maintainers received a stream of half-polished patches.

Our current cadence is simpler and more respectful. On a regular schedule, someone reviews the inventory, checks the state of related upstream issues and PRs, reruns the upstream diff against the current pin, updates statuses, and only then decides which entries are ready to submit.

This sounds slow, but it is faster in aggregate. One clean, well-scoped submission does more good than several rushed ones that stall in review.

The cadence has a second benefit: it is when retirements actually happen. Without a recurring pass, "retire this once upstream merges" becomes a promise nobody keeps.

## How it lands in MegaCpp

Production MegaCpp inherits the patch lane, but the shape changes.

First, the import-time patch surface shrinks. A fast-moving research-stack can justify small feature-detected overlays; long-lived product code should move those decisions into clearer seams such as subclassing, configuration, or explicit integration points.

Second, the environment matrix narrows. Production ships fewer hardware lanes than research, and the difference between research and production stacks is tracked explicitly.

Third, release notes absorb the retirement trail. A production release should make it obvious which local patches disappeared and which upstream versions replaced them.

## Ablations and what we kept

We tried three things that did not survive contact with real training.

We tried a single lockfile across CUDA, TPU, and CPU. It was painful to produce, painful to update, and inaccurate often enough that it became a source of confusion rather than clarity. We dropped it in favor of explicit pinned environments.

We tried to carry every Megatron fix as a full fork. That worked, but rebasing against a moving development branch became too expensive for changes that were often very small. We kept fuller forks where the diff justified them and used lighter overlays elsewhere.

We also tried a strict upstream-first rule: never land a local patch until a polished upstream PR exists. That sounds principled, but on a fast-moving training stack it can waste an entire training window. We replaced it with a better rule: fix locally when needed, draft the upstream-quality explanation immediately, and submit when the patch is ready.

What we kept is the part that compounds: pinned environments, a small set of local forks or overlays, a public-facing patch inventory, a regular upstream diff, and a weekly merge-back cadence. We also kept three hard rules: every local patch must be tracked, every tracked patch needs a retirement condition, and nothing retires silently.

## Production checklist

- Pinned environments are the source of truth. No lockfile heroics, and installs should not drift underneath you.
- Carry full forks only when the diff is substantial; use lighter overlays for genuinely small surgical fixes.
- Every local patch needs a matching public-facing entry with a reproducer and a readable explanation.
- Every entry needs a retirement condition and a clear upstream status.
- Regular upstream-diff checks should make undocumented drift impossible to ignore.
- Weekly merge-back reviews should update status and retire work that upstream has absorbed.
- Import-time patches should be idempotent and feature-detect the upstream shape they compensate for.
- Retirement should be explicit in version control, not tribal knowledge.
- Submission volume should respect reviewer bandwidth.

## References

- [MegaCpp source repository](https://github.com/DatasunriseOU/cppmega)
- [MegaCpp sample pack](https://github.com/DatasunriseOU/site_samples)
- [PyTorch previous versions guide](https://pytorch.org/get-started/previous-versions/)
- [PyPA packaging guide](https://packaging.python.org/en/latest/)
