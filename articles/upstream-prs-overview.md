---
title: "Upstream PRs: how a small training shop ends up patching everyone else's libraries"
description: "A guided tour of the upstream contributions we are submitting back to the open-source training stack, the cadence we hold ourselves to, and the categories that keep showing up."
date: "2026-04-18"
tags: ["upstream", "open-source", "megatron", "tilelang", "mamba", "liger"]
---

If you train a frontier-shaped model on a stack made of Megatron-LM, TileLang, Liger-Kernel, `state-spaces/mamba`, and several NVIDIA reference kernels, you will sooner than you expect become the de facto maintainer of all of them. Not officially, but in the sense that the bugs you hit are real, the maintainers may not have hit them yet, and your training run does not get to wait for the upstream sprint cycle. This post is the honest tour of the upstream PR pipeline we keep open: why it exists, the checklist before filing, the categories the work falls into, and the cadence we settled on.

## Why this matters

The selfish reason first: every patch we carry locally is a future merge conflict. We pin active upstream revisions, and we sometimes carry a small set of open upstream patches on top. Those branches move every week, and every patch we never upstreamed has to be manually rebased by someone who still remembers why it exists. The half-life of "I'll write it up later" is roughly one model preset.

The less selfish reason is that downstream users benefit when the rest of the ecosystem does the same thing. The DSA indexer memory work that landed in TensorRT-LLM unblocked inference planning before the same patch had to be carried locally. The TileLang `LowerBulkCopy` warn-and-fallback is also part of why the Mamba3 MIMO backward kernels compile under TMA lowering at all. This arrangement only works long-term if teams that hit a bug write it up properly.

The third reason is calibration. Writing an upstream PR forces you to separate "broken in our integration" from "broken in the library". A surprising number of bugs evaporate at that boundary. Of the sixteen packs in the current queue, several turned out to be already fixed upstream or aimed at the wrong upstream surface once we re-checked them carefully. The checklist below exists because of that.

## What goes into a submission pack

A pack is the bundle of artifacts we need to file one upstream contribution. Our filing checklist is the source of truth for whether a pack is ready, and it has the following shape.

First, a markdown template for the issue or PR body. It is written in English, avoids non-public infrastructure or branch labels, and cites only the public identifiers needed to explain the bug.

Second, a self-contained reproducer. The reproducer must run as `python reproducer.py` against a clean checkout of the target library at a named SHA, with the dependency versions pinned next to it. It must print a `BUG_REPRODUCED` (or equivalent) sentinel when it triggers the bug, and a `FIX_VALIDATED` sentinel when run against the patched code. The reproducer also stamps the host capability in its first line of output so the maintainer can tell at a glance whether their own machine should reproduce.

Third, a validation-manifest entry. This records which host last validated the reproducer, what the exit code was, and which sentinels were printed. The manifest is the only thing we trust when we ask "is this pack ready"; we do not trust the date in the markdown body, and we do not trust anyone's memory.

Fourth, the upstream-state field. Before filing, the target project gets searched for relevant issues and pull requests, and the result is recorded as a new report, an overlap with an existing thread, or something already fixed. If it is already fixed, the pack does not get filed; it gets repurposed as regression coverage. The TileLang `LowerBulkCopy` 3D shared-memory case is exactly that story: by the time it was re-checked, the relevant warn-and-fallback work had already shipped, so the reproducer stayed as regression coverage instead of becoming a duplicate bug report.

Fifth, an explicit "post it" gate. No pack, however ready, is filed without a human typing the words. We never automated past that gate.

## The checklist we actually run

The pre-flight is short on purpose, because long checklists do not get followed. In order:

1. The reproducer passes today on a host we can name. Not last week. Not "the manifest says it passed". Today.
2. The template markdown renders cleanly in GitHub's preview pane. Code fences survive, tables survive, and no non-public links bleed through.
3. The reproducer is attached as a file or a gist, not pasted inline. Hundreds of lines of Python in an issue body burns reviewer attention.
4. No non-public-only language: no infrastructure references, no non-public branch codes, no employee names other than the named authors.
5. For Megatron PRs we run the project's required formatter before the diff goes in, because the upstream PR template makes that a hard checklist item.
6. The filing approval is recorded by the people doing the work.

The checklist has caught real things. The FP8 dispatch hazard for SparseMLA was originally aimed at TileLang because that is where the kernel lives. Reading the body in preview made it obvious the bug is actually in the Transformer Engine `Float8Tensor` wrapper (`.dtype` lies, `.data_ptr()` returns NULL, `.contiguous()` does not unwrap), so the target shifted accordingly. Similarly, an early manifest pointed one Mamba backward note at the wrong repository even though the public sample being changed lived elsewhere. Both targeting errors were fixed before anything was filed.

## The cadence

We submit in waves, not as a stream. A wave is two to four packs filed within a small window, batched by target repository, then a two-to-three-day pause before the next wave. The reason is simple: maintainers have inboxes. If six issues land on the same repository on the same morning, two may get triaged and the other four may sit. If two land, both are more likely to get triaged.

Within a wave we follow a coarse priority order. First come defensive fixes against bugs that crash training, such as the DSA CUDA-graph capture issue or the Megatron `Float16Module` cast that breaks Mamba3's fp32 contract. Second come fixes that can piggyback on an already-open upstream pull request, where a comment is more useful than a competing patch. Third come larger refactors that need real maintainer attention, such as SparseMLA dimension generalization or the Mamba3 MIMO 3D-to-2D shared-memory refactor. Fourth come bug reports for which there is not yet a fix to offer, such as the TileLang FloorMod divide-by-zero in `LayoutInference`.

The cadence rule that takes the longest to internalize is that "ready" does not mean "filed today". A pack can sit in the ready state for a week while waiting for the right wave; the cost of holding a ready pack is low, and the cost of dumping six issues on one maintainer is not.

## The categories

The packs cluster into a small number of recurring shapes. Once we noticed the shapes, the writing got faster, because each shape has a template.

The first category is **CUDA-graph and graph-capture safety**. Library code from a year ago routinely contains `torch.equal(...)`, `tensor.any()`, or `if torch.any(idx < 0)` constructs that implicitly `cudaStreamSynchronize` and crash graph capture with `cudaErrorStreamCaptureUnsupported`. The fix is always the same: gate validations on `torch.cuda.is_current_stream_capturing()`, or rewrite the branchy logic into a branchless clamp/scatter/fixup. Pack 01 (DSA in Megatron-LM) is the canonical example.

The second category is **dispatch hazards introduced by tensor-wrapper types**. Float8Tensor, QuantizedTensor, and any other `__torch_dispatch__`-based wrapper has a habit of looking like a normal tensor at the Python level (`.dtype`, `.shape`, `.contiguous()` all behave) while being unsafe to hand to a raw CUDA or TileLang kernel. Pack 03 is the FP8 SparseMLA case. The Megatron `Float16Module` blanket bf16 cast in pack 16 is the same family in reverse: an upstream "helper" silently rewrites tensor dtypes that another library expects to keep in fp32, and the result is either a clean dispatch error or silent NaN.

The third category is **kernel-side numerical and dimension correctness**. The SparseMLA dimension generalization (pack 02), the SparseMLA backward `accum_dtype` precision fix (pack 14), and the missing GQA branch in Mamba3 MIMO backward (pack 05) sit here. None is glamorous; they are all either "this kernel hardcoded one shape and crashes on every other shape" or "this buffer was bf16 where it needed fp32 and the gradient drifted".

The fourth category is **memory-footprint reductions**. The DSA `_compute_index_scores` per-head streaming accumulator (pack 12) is the loudest: an `einsum` that materialized a 16 GiB intermediate, replaced with a per-head `bmm` that reuses a 268 MiB output buffer. Math unchanged, working set ~60x smaller. These patches almost always overlap with at least one in-flight upstream PR, which is why they go in as comments rather than competing PRs.

The fifth category is **integration and dispatcher gaps in Megatron-LM**. The Hopper FLCE dispatcher (pack 10) crashes with `ValueError: Unsupported architecture: 9` on every cc!=10 device because the Blackwell entry was the only branch wired in. The Mamba `LinearCrossEntropyModule` wiring (pack 11) was correctly added in one PR and silently reverted three weeks later by a rebase-miss in another. These are the easiest packs to write and the hardest to file, because the right framing is "your CI did not catch this".

The sixth category is **toolchain and compiler bugs in TileLang**. One example is the FloorMod divide-by-zero in `LayoutInference` when TMA lowering is enabled on Mamba3 backward kernels. Another is the now-fixed `LowerBulkCopy InputDim==2` assert, which remains useful as a regression guard. These are bug reports, not patch drops; the right fix lives inside TVM's iter-map normalizer.

The seventh category is **legality-preserving refactors that unlock a lowering path**. Pack 07 (Mamba3 MIMO backward 3D->2D smem flatten) is the entire category. Every `[c, r1, r2]` indexer becomes `[c, r1*R + r2]`; smem footprint and register pressure are identical and gradients are bitwise-equal to the unflattened baseline within bf16 rounding. The reason to do it is that TileLang's TMA bulk-copy lowering requires `InputDim()==2`; once the descriptors are 2D, the backward kernel becomes eligible for TMA pipelining on Hopper.

## Honest about state

None of the sixteen packs has been filed yet. Some are still blocked on fresh repro evidence. The SparseMLA precision fix remains a code-level note without a checked-in reproducer bundle, so it does not yet meet the bar described here. The `Float16Module` cast note shares reproducer coverage with another runtime issue but would still be tracked separately if filed. The `LowerBulkCopy` note has already been re-classified from "issue to file" to "local regression guard" because the underlying fix has shipped upstream.

These packs are written alongside the training work rather than by a dedicated open-source liaison. The packs that get written are the ones whose absence would cost more rebase time than the writeup costs. That filter is blunt, and also why the packs that do exist are concrete enough for someone else to validate.

## Production checklist

- Every patch we carry locally gets either an upstream pack or an explicit decision not to file, recorded in the filing checklist.
- Every pack has a reproducer that runs against a named upstream SHA and prints a sentinel.
- Reproducers stamp the host capability and the dependency versions in their first lines of output.
- The validation manifest is the source of truth for "is this pack ready", not the markdown body.
- We file in waves of two to four packs, batched by target repository, with two to three days between waves.
- We never open a competing PR against an open upstream PR; we comment on the existing thread instead.
- We do not file packs that are already fixed upstream; we repurpose them as regression tests.
- The filing decision is always made by a human and recorded.
- No non-public infrastructure references, no non-public branch codes, and no employee names other than the named authors.
- For Megatron PRs we run the project's formatter before any diff is attached.

## References

- [GitHub pull request documentation](https://docs.github.com/en/pull-requests)
- Selected upstream pull requests and issues in TileLang, Megatron-LM, Liger-Kernel, and TensorRT-LLM that match the categories described above.
