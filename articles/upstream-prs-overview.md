---
title: "Upstream PRs: how a small training shop ends up patching everyone else's libraries"
description: "A guided tour of the upstream contributions we are submitting back to the open-source training stack, the cadence we hold ourselves to, and the categories that keep showing up."
date: "2026-04-18"
tags: ["upstream", "open-source", "megatron", "tilelang", "mamba", "liger"]
---

If you train a frontier-shaped model on a stack made of Megatron-LM, TileLang, Liger-Kernel, `state-spaces/mamba`, and several NVIDIA reference kernels, you will sooner than you expect become the de facto maintainer of all of them. Not officially, but in the sense that the bugs you hit are real, the maintainers may not have hit them yet, and your training run does not get to wait for the upstream sprint cycle. This post is the honest tour of the upstream PR pipeline we keep open: why it exists, the checklist before filing, the categories the work falls into, and the cadence we settled on.

## Why MegaCpp cares about this

The selfish reason first: every patch we carry locally is a future merge conflict. We pin active upstream revisions, and we sometimes carry a small set of open upstream patches on top. Those branches move every week, and every patch we never upstreamed has to be manually rebased by someone who still remembers why it exists. The half-life of "I'll write it up later" is roughly one model preset.

The less selfish reason: we benefit from the rest of the ecosystem doing the same thing. The DSA indexer memory work that landed as TensorRT-LLM PR #12198 unblocked our inference plans before we had to write the same patch ourselves. The TileLang `LowerBulkCopy` warn-and-fallback in PR #746 is the only reason our Mamba3 MIMO backward kernels compile under TMA lowering at all. We are downstream of all of these projects, and that arrangement only works long-term if everyone who hits a bug writes it up properly.

The third reason is calibration. Writing an upstream PR forces you to separate "broken in our integration" from "broken in the library". A surprising number of bugs evaporate at that boundary. Of the sixteen packs in the current queue, several turned out to be already fixed upstream or aimed at the wrong upstream surface once we re-checked them carefully. The checklist below exists because of that.

## What goes into a submission pack

A pack is the bundle of artifacts we need to file one upstream contribution. Our filing checklist is the source of truth for whether a pack is ready, and it has the following shape.

First, a markdown template for the issue or PR body. It is written in English, avoids non-public infrastructure or branch labels, and cites only the public identifiers needed to explain the bug.

Second, a self-contained reproducer. The reproducer must run as `python reproducer.py` against a clean checkout of the target library at a named SHA, with the dependency versions pinned next to it. It must print a `BUG_REPRODUCED` (or equivalent) sentinel when it triggers the bug, and a `FIX_VALIDATED` sentinel when run against the patched code. The reproducer also stamps the host capability in its first line of output so the maintainer can tell at a glance whether their own machine should reproduce.

Third, a validation-manifest entry. This records which host last validated the reproducer, what the exit code was, and which sentinels were printed. The manifest is the only thing we trust when we ask "is this pack ready"; we do not trust the date in the markdown body, and we do not trust anyone's memory.

Fourth, the upstream-state field. Before we file, we search the target repo for relevant issues and PRs and record whether the pack is a new report, overlaps an existing thread, or is already fixed. If it is already fixed, the pack does not get filed; instead it gets repurposed as regression coverage. Pack 08 (the TileLang `LowerBulkCopy` 3D smem case) is exactly that story: when we first wrote it the assert was still in the tree, and by the time we re-checked, PRs #746, #761 and #2005 had already shipped a warn-and-fallback, so we kept the reproducer as regression coverage instead of filing a duplicate bug.

Fifth, an explicit "post it" gate. No pack, however ready, is filed without a human typing the words. We never automated past that gate.

## The checklist we actually run

The pre-flight is short on purpose, because long checklists do not get followed. In order:

1. The reproducer passes today on a host we can name. Not last week. Not "the manifest says it passed". Today.
2. The template markdown renders cleanly in github.com's preview pane. Code fences survive, tables survive, no private-only links bleed through.
3. The reproducer is attached as a file or a gist, not pasted inline. Hundreds of lines of Python in an issue body burns reviewer attention.
4. No non-public-only language: no infrastructure references, no non-public branch codes, no employee names other than the named authors.
5. For Megatron PRs we run the project's required formatter before the diff goes in, because the upstream PR template makes that a hard checklist item.
6. The filing approval is recorded by the people doing the work.

The checklist has caught real things. Pack 03 (the FP8 dispatch hazard for SparseMLA) was originally aimed at the TileLang repo because that is where the kernel lives. Reading the body in preview made it obvious the bug is actually in the Transformer Engine `Float8Tensor` wrapper (`.dtype` lies, `.data_ptr()` returns NULL, `.contiguous()` does not unwrap), and the right repo is `NVIDIA/TransformerEngine`. Similarly, pack 07 had its target listed as `tile-ai/tilelang` in an early manifest; the file being changed is the public Mamba backward kernel sample in `state-spaces/mamba`, and TileLang only enters as the lowering target. Both manifest errors were fixed before either issue existed.

## The cadence

We submit in waves, not as a stream. A wave is two to four packs filed within a small window, batched by target repo, then a two-to-three-day pause before the next wave. The reason is simple: maintainers have inboxes. If we drop six issues on the same repo on the same morning, two of them get triaged and the other four sit. If we drop two, both get triaged.

Within a wave we follow a coarse priority order. First, defensive fixes against bugs that crash training (the DSA CUDA-graph capture issue, the Megatron `Float16Module` cast that destroys Mamba3's fp32 contract). These are the easy reviews and they build credibility for the harder ones. Second, fixes that piggyback on an already-open upstream PR (our DSA `_compute_index_scores` memory pack is offered as a comment on the open Fused Indexer Loss Kernel PR rather than as a competing PR; opening a competitor would stall both). Third, the larger refactors that need real maintainer attention (the SparseMLA dimension generalization, the Mamba3 MIMO 3D-to-2D smem refactor). Fourth, bug reports for which we do not have a fix to offer yet (the TileLang FloorMod divide-by-zero in `LayoutInference`, where we have a clean reproducer but the right fix lives inside TVM's iter-map normalizer and we are not the right authors).

The cadence rule that took us longest to internalize is that "ready" does not mean "filed today". A pack can sit at `ready=Y` for a week while we wait for the right wave; the cost of holding a ready pack is zero, the cost of dumping six issues on a maintainer is not.

## The categories

The packs cluster into a small number of recurring shapes. Once we noticed the shapes, the writing got faster, because each shape has a template.

The first category is **CUDA-graph and graph-capture safety**. Library code from a year ago routinely contains `torch.equal(...)`, `tensor.any()`, or `if torch.any(idx < 0)` constructs that implicitly `cudaStreamSynchronize` and crash graph capture with `cudaErrorStreamCaptureUnsupported`. The fix is always the same: gate validations on `torch.cuda.is_current_stream_capturing()`, or rewrite the branchy logic into a branchless clamp/scatter/fixup. Pack 01 (DSA in Megatron-LM) is the canonical example.

The second category is **dispatch hazards introduced by tensor-wrapper types**. Float8Tensor, QuantizedTensor, and any other `__torch_dispatch__`-based wrapper has a habit of looking like a normal tensor at the Python level (`.dtype`, `.shape`, `.contiguous()` all behave) while being unsafe to hand to a raw CUDA or TileLang kernel. Pack 03 is the FP8 SparseMLA case. The Megatron `Float16Module` blanket bf16 cast in pack 16 is the same family in reverse: an upstream "helper" silently rewrites tensor dtypes that another library expects to keep in fp32, and the result is either a clean dispatch error or silent NaN.

The third category is **kernel-side numerical and dimension correctness**. The SparseMLA dimension generalization (pack 02), the SparseMLA backward `accum_dtype` precision fix (pack 14), and the missing GQA branch in Mamba3 MIMO backward (pack 05) sit here. None is glamorous; they are all either "this kernel hardcoded one shape and crashes on every other shape" or "this buffer was bf16 where it needed fp32 and the gradient drifted".

The fourth category is **memory-footprint reductions**. The DSA `_compute_index_scores` per-head streaming accumulator (pack 12) is the loudest: an `einsum` that materialized a 16 GiB intermediate, replaced with a per-head `bmm` that reuses a 268 MiB output buffer. Math unchanged, working set ~60x smaller. These patches almost always overlap with at least one in-flight upstream PR, which is why they go in as comments rather than competing PRs.

The fifth category is **integration and dispatcher gaps in Megatron-LM**. The Hopper FLCE dispatcher (pack 10) crashes with `ValueError: Unsupported architecture: 9` on every cc!=10 device because the Blackwell entry was the only branch wired in. The Mamba `LinearCrossEntropyModule` wiring (pack 11) was correctly added in one PR and silently reverted three weeks later by a rebase-miss in another. These are the easiest packs to write and the hardest to file, because the right framing is "your CI did not catch this".

The sixth category is **toolchain and compiler bugs in TileLang**. Pack 13 is the FloorMod divide-by-zero in `LayoutInference` when TMA lowering is enabled on Mamba3 backward kernels. Pack 08 is the now-fixed `LowerBulkCopy InputDim==2` assert that we keep around as a regression guard. These are bug reports, not PRs; the right fix lives inside TVM's iter-map normalizer and we trust the TileLang maintainers to land it correctly.

The seventh category is **legality-preserving refactors that unlock a lowering path**. Pack 07 (Mamba3 MIMO backward 3D->2D smem flatten) is the entire category. Every `[c, r1, r2]` indexer becomes `[c, r1*R + r2]`; smem footprint and register pressure are identical and gradients are bitwise-equal to the unflattened baseline within bf16 rounding. The reason to do it is that TileLang's TMA bulk-copy lowering requires `InputDim()==2`; once the descriptors are 2D, the backward kernel becomes eligible for TMA pipelining on Hopper.

## Honest about state

None of the sixteen packs has been filed yet. Some are blocked on a fresh repro (pack 10 needs a clean H200 receipt against an unpatched tree). Pack 14, the SparseMLA precision fix, is a code-level note without a checked-in reproducer bundle, so it does not yet meet our own bar. Pack 16 (the Megatron `Float16Module` cast) shares its reproducer with pack 05 but goes in as a separate issue against `NVIDIA/Megatron-LM`. Pack 08 has been re-classified from "issue to file" to "local regression guard" because the underlying fix already shipped in TileLang PR #746.

David Gornshtein and Boris Tamarkin write these packs alongside the training work. We do not have a dedicated open-source liaison. The packs that get written are the ones whose absence would cost us more rebase time than the writeup costs. That filter is uncomfortably honest, and also why the packs that do exist are concrete enough for someone else to validate.

## Production checklist

- Every patch we carry locally gets either an upstream pack or an explicit decision not to file, recorded in the filing checklist.
- Every pack has a reproducer that runs against a named upstream SHA and prints a sentinel.
- Reproducers stamp the host capability and the dependency versions in their first lines of output.
- The validation manifest is the source of truth for "is this pack ready", not the markdown body.
- We file in waves of two to four packs, batched by target repo, with two to three days between waves.
- We never open a competing PR against an open upstream PR; we comment on the existing thread instead.
- We do not file packs that are already fixed upstream; we repurpose them as regression tests.
- The filing decision is always made by a human and recorded.
- No non-public infrastructure references, no non-public branch codes, and no employee names other than the named authors.
- For Megatron PRs we run the project's formatter before any diff is attached.

## References

- Filing checklist and validation manifest used to track readiness.
- TileLang PR #746 ([Refactor] Merge bulk copy and improve layout inference) and follow-ups #761, #2005.
- Megatron-LM PR #3345 (Hopper FLCE kernels), PR #3226 (LinearCrossEntropyModule wiring), PR #3207 (MTP replay), PR #3674 (DSA absorbed MLA + TileLang fused), PR #4039 (Fused Indexer Loss Kernel), PR #4268 (delayed wgrad overlap).
- Liger-Kernel issue #968 (closed) and PRs #1126 (draft assertion) / #1182 (reduction plumbing).
- TensorRT-LLM PR #12198 as the inference-side analogue of pack 12.
