---
title: "Torch/XLA 2.11 expectations vs TPU reality"
description: "What MegaCpp expected from the Torch/XLA 2.11 line on TPU, what the shipped stack actually looked like in practice, and how that changed our bringup strategy."
date: "2026-04-19"
author: "David Gornshtein"
tags: ["tpu", "xla", "PyTorch", "torch-xla", "bringup", "MegaCpp"]
---

When teams talk about a new Torch/XLA line, they often compress two very different questions into one. The first question is what the version number suggests: newer runtime, newer PJRT plumbing, maybe better cache behavior, maybe fewer bringup patches. The second question is the one operators actually have to answer: what exact wheel lineage, `libtpu` build, cache path, and compile policy can survive a real TPU training lane?

MegaCpp ended up learning that the second question mattered much more. The public repo history shows an initial expectation that the newer Torch/XLA 2.11-class stack might simplify TPU work. What actually shipped into our working TPU lane was more complicated: custom builds, mixed-stack experimentation, partial wins, and a bringup strategy that had to stay stricter than the version number suggested.

## What we expected from the 2.11 line

The expectation was reasonable. A newer Torch/XLA line appeared to offer three things at once:

- a newer OpenXLA and PJRT contract
- a chance to use newer `libtpu` builds
- fewer local workarounds around TPU compile and startup behavior

That hope shows up indirectly in the repo history. There is a dedicated March 2026 fix titled `Guard FSDP nightly mesh fix on torch 2.11`, which already tells you the team was actively validating 2.11-specific behavior rather than treating TPU as frozen on an older stack. The changelog also records a full TPU deployment wave where all eight workers on one TPU experiment were moved to `torch 2.11.0a0+git7afdbae`, `torch_xla 2.11.0+gitc04e61c`, `libtpu 0.0.37`, and `jax 0.9.0`.

On paper, that looks like the kind of stack refresh that should collapse some bringup complexity. A newer framework line plus a newer TPU runtime is exactly the combination people usually hope will reduce patch debt.

## What Google actually shipped into the practical path

The operator docs and install scripts show a different reality. MegaCpp's checked-in TPU install path does not say "use the stock upstream 2.11 wheels and move on." It says the repo-preferred path is a custom wheel stack. The mainline TPU installer pins:

- custom `torch 2.9.0a0+git21fec65`
- custom `torch_xla 2.9.0+gitc04e61c`
- Python `3.13`
- `jax 0.9.0`
- `libtpu 0.0.24` minimum, with `0.0.36` as the preferred tested receipt

The repo README says this explicitly: the validated TPU lane uses the custom 2.9-based stack, and the custom `torch_xla` line is required for the modern `libtpu` and PJRT contract because stock PyPI `torch_xla==2.9.0` stayed constrained to an older `libtpu` lane.

That is the first important correction to the simple 2.11 story. The practical MegaCpp TPU path did not converge on "2.11 solved it." It converged on "we needed custom wheels to get the TPU runtime contract we actually wanted, and the stable operator story remained a custom stack."

## The 2.11 experiments were real, but they did not become the simple default

The public changelog makes clear that 2.11 was not imaginary or incidental. It records a special validation lane with a newer OpenXLA snapshot, newer PJRT framework version, and custom `torch_xla 2.11` wheels. It also records why that newer line did not instantly become the whole TPU story.

One operator note is especially revealing: the fleet became intentionally mixed-stack. Some TPU machines stayed on the custom 2.9 line, while a validation lane ran a newer 2.11 build. The changelog warns that comparisons had to remain explicit because the mixed stack could not be treated as one uniform environment. That is the opposite of a clean version-bump narrative.

Another note is even more operationally important: the validation lane's custom HLO cache patch still did not remove the expensive `libtpu` compilation step. The repo notes that cache files were written, but restart still re-entered `libtpu Compile()`. In other words, part of the expected 2.11 story was "newer stack, maybe persistent cache finally pays off." The observed story was "some cache plumbing improved, but the operator-facing compile pain remained."

There is a second correction here. "Google shipped a newer TPU software line" and "the practical TPU bringup got much easier" are not the same statement.

## Why the compile-cache story mattered so much

This detail shaped bringup more than any headline version number. MegaCpp's TPU notes repeatedly narrow the real runtime contract to a few stubborn truths:

- TPU launch must use the PJRT TPU runtime path explicitly
- SPMD must be enabled early and treated as part of startup correctness
- `XLA_NO_SPECIAL_SCALARS=1` remains part of the core run contract
- the compilation cache needs an early, explicit local path such as `XLA_COMPILATION_CACHE_DIR`
- model-level `torch.compile(...)` is not the TPU strategy; the practical path is `torch_xla.compile()` around forward and backward micro-steps, with separate optimizer-step handling

That is a much more hands-on contract than the optimistic reading of a newer framework line. The TPU setup guide also warns against stale habits such as exporting `XLA_USE_BF16=1` for the current Pallas attention path or assuming that `XLA_USE_SPMD=1` is the real activation switch. The working lane is defined by current repo code and startup order, not by folklore about what XLA used to want.

This is why the cache story hurt so much. If persistent cache had fully delivered on restart, bringup would have become much less fragile operationally. Instead, the public notes show a more limited outcome: cache write paths existed, but restart behavior still did not eliminate the expensive compile wall in the way operators hoped.

## How this changed our TPU bringup strategy

The version story stopped being the organizing principle. Bringup moved to a stricter playbook.

### 1. We stopped treating upstream version labels as the main unit of truth

The operator docs now privilege the checked-in install script, current runtime notes, and training code over generic assumptions about a release line. That is the right response when "2.11" can mean several materially different combinations of OpenXLA snapshot, `libtpu` version, wheel provenance, and startup flags.

### 2. We separated "experimental validation lane" from "operator-preferred lane"

This is one of the cleanest decisions in the repo. The public docs distinguish the validated day-to-day TPU stack from older or alternative reference paths. That separation matters because it stops the team from over-reading one successful experiment as a global convergence point.

### 3. We treated bringup as a runtime-contract problem, not a package-upgrade problem

The current TPU documentation is almost entirely about contract details: compile mode, retry ladder, flag profiles, cache location, sharding shape, and startup order. That is exactly what you would expect after learning that a newer framework line does not automatically erase TPU-specific operational constraints.

### 4. We kept mixed-stack evidence explicit instead of flattening it

The changelog explicitly warns that 2.9 and 2.11 comparisons must stay run-by-run. That is a healthy discipline. Once one lane has a newer OpenXLA snapshot and another does not, a good bringup log should not pretend they are directly interchangeable.

## The deeper lesson

The main lesson is not that Torch/XLA 2.11 was bad. The lesson is that TPU bringup quality depends less on the abstract release name than on the exact runtime contract that the team can reproduce.

For MegaCpp, the public record shows that a newer 2.11 line was useful as an experimental and validation surface. It exposed version-specific fixes, newer OpenXLA behavior, and a place to test cache-related ideas. But the practical operator lane still centered on a custom, pinned stack and on repo-specific startup discipline. In that sense, the 2.11 expectation was "maybe the platform line itself becomes the simplifier." The shipped reality was "the simplifier is still a checked-in install path plus a narrow, reproducible runtime contract."

That is a more conservative story, but it is also the more useful one. It explains why TPU bringup eventually became less about chasing a magic wheel version and more about preserving a stable launch recipe, recording exact receipts, and separating hoped-for platform behavior from observed platform behavior.

## What this means for future TPU stack upgrades

The best way to read a future Torch/XLA upgrade is now obvious.

- Ask which exact wheel lineage is operator-blessed.
- Ask which `libtpu` line is actually validated with it.
- Ask whether restart really reuses compilation work or only saves tracing overhead.
- Ask whether the sharding and compile policy stayed the same.
- Ask whether the new line replaced the old lane or merely created a second experimental lane.

Those questions are better than "are we on 2.11 yet?" because they are the ones that determine whether TPU bringup gets easier in practice.

## References

- [MegaCpp sample pack](https://github.com/DatasunriseOU/site_samples)
- [PyTorch/XLA runtime guide](https://docs.pytorch.org/xla/master/runtime.html)
- [PyTorch/XLA PJRT guide](https://docs.pytorch.org/xla/master/learn/pjrt.html)
- [PyTorch/XLA recompilation guide](https://docs.pytorch.org/xla/master/perf/recompilation.html)
- [Cloud TPU v6e overview](https://cloud.google.com/tpu/docs/v6e)
- [Cloud TPU performance and training docs](https://cloud.google.com/tpu/docs/v6e-training)
