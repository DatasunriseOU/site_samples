---
title: "Training speed anatomy on H200"
description: "What actually sets training speed on H200 in the current prototype and MegaCpp receipts: compile warmup policy, block mix, memory shape, and why local wins often fail to move whole-step throughput."
date: "2026-04-18"
tags: ["h200", "training", "performance", "nam52", "nam56r", "moe", "mamba"]
---

**TL;DR:** H200 training speed in the current stack is shaped less by any single kernel headline and more by step anatomy: compile policy, block mix, communication overlap, memory shape, and whether a supposed fast path is even active. The strongest current CUDA lane is dense H200, but the repo's own reports show that regional compile plus MoE can still stall in explicit warmup, attention can be a small minority of total step time, and some of the biggest gains come from removing unnecessary allocations rather than from rewriting the hottest-looking kernel.

People like to ask for a single answer to "what makes H200 fast?" The useful answer is structural. Training speed is the sum of setup costs, compile behavior, forward and backward mix, communication policy, and memory pressure. On a hybrid stack like NAM52 or NAM56R, those pieces vary by pattern. That is why the best H200 notes in the repo are not generic benchmark blurbs; they are receipts tied to exact lanes and exact code paths.

## Start with the lane, not the GPU

The current README already frames H200 correctly. It says the strongest current CUDA lane is dense H200, while other hardware lines remain partial or experimental. That wording matters because it refuses to treat "H200" as a single performance fact. A dense lane, a MoE lane, and a hybrid Mamba lane can all run on the same accelerator and still have very different speed anatomy.

The changelog reinforces that point. One audit note says attention was only around 6 percent of step time in a particular regional-compile configuration with `head_dim=64/128`. That is a deceptively important number. It means many attention-centric optimization claims are bounded before they start. Even a perfect attention win cannot move total step speed much if attention is a thin slice of the step.

| Layer of analysis | Wrong question | Better question |
| --- | --- | --- |
| hardware | "How fast is H200?" | "Which H200 lane and which pattern?" |
| kernels | "Did attention get faster?" | "How much of the step is attention on this lane?" |
| compile | "Did compile complete?" | "Did explicit warmup help or did it stall the lane?" |
| memory | "Did peak memory drop?" | "Did the drop enable a larger or more stable training configuration?" |

This framing sounds obvious, but it filters out most low-value performance discussion immediately.

## Compile policy is part of speed, not just startup overhead

The March H200 regression report is one of the clearest documents in the codebase because it isolates a concrete bottleneck: explicit CUDA compile warmup on `regional_compile + MoE` lanes. The report says the current regression is not well explained by blaming a newer PyTorch alone. Instead, the strongest repo-side regression sits in compile-warmup policy.

The historical context matters. Earlier fixes were already in place, including a fused RoPE batch-layout guard and an `EBlock` regional-compile skip. But later logic re-enabled explicit compile warmup under the assumption that the earlier fix had removed the dangerous crash class. The live repros disproved that assumption for current H200 regional-compile plus MoE lanes.

The report then walks through three useful receipts.

1. A stalled warmup lane with warmup enabled still gets stuck in compile warmup even with effective MoD bypass.
2. A manual no-warmup workaround lane skips warmup, keeps caches growing lazily, and progresses into autotune and MoE forward work.
3. A patched default lane adopts the same practical behavior automatically by skipping explicit warmup on CUDA `regional_compile + moe_enabled` paths.

This is a strong lesson because it shows how startup policy changes total runtime behavior. Compile warmup is often described as a harmless front-loaded cost that pays off later. Here it was an execution blocker. On this class of H200 lanes, skipping explicit warmup was not a micro-optimization. It was the difference between stalling in setup and reaching the real training path.

```text
if device == cuda and regional_compile and moe_enabled:
    compile_warmup = skipped
else:
    compile_warmup = normal_policy
```

That simple rule is more valuable than a dozen vague claims about "compiler stability."

## Whole-step speed comes from dominant work, not from the most fashionable kernel

Once the lane gets past setup, the next question is where the time actually goes. The repo provides several grounded examples that push against hype-driven optimization.

The changelog note about attention being only a small part of step time is one. Another comes from the MegaCpp Mamba linear cross-entropy reproducer. That document shows a case where restoring class parity on the Mamba output layer removes a large unnecessary logits allocation and turns an OOM-prone high-end run into a stable one. The gain is not from magical new math. It comes from eliminating an avoidable memory shape that was dragging the training path down.

The DSA reproducer tells a similar story from another direction. The fused version avoids materializing a giant intermediate and keeps peak memory much lower while preserving correctness. On paper that is a memory optimization; in practice it is also a speed optimization whenever the original intermediate is driving allocator churn, launch instability, or configuration limits.

The lesson is straightforward. Kernel-level work matters, but whole-step speed on H200 is often governed by:

- whether the kernel is on the active path
- how much of the step that path occupies
- whether memory shape is forcing smaller microbatches or triggering instability
- whether communication or compile policy is the real limiter

That is why local profiler wins frequently fail to move the end-to-end number. They solve the wrong slice.

## Communication and overlap still matter, but only when the lane can use them

The current changelog also records several communication-side changes with real performance implications: Megatron-style bucket handling, overlap-related harness patterns, and fp32 gradient-reduction support in the optimizer path. These are meaningful because they alter how much of the step is hidden behind communication and how stable the reduction path remains.

But once again, the repo is careful not to overclaim. Some items are explicitly called out as no-ops or partial truths. Shared expert overlap is not treated as a free speedup if the concurrency path is not really there. Router dtype claims are checked against autocast reality. And several deferred items are named honestly instead of being retroactively counted as delivered throughput work.

This is exactly how H200 performance work should be reported. If overlap exists only on paper, do not count it. If a precision policy helps only one branch of the model, say so. If a parity gap means a supposedly optimized path is not running in the current recipe, then the right answer is not "performance is disappointing" but "the fast path is not actually active."

| Speed lever | When it helps | When it disappoints |
| --- | --- | --- |
| comm bucket tuning | communication-bound or overlap-friendly lanes | compute-dominant lanes where comm is already hidden |
| fp32 grad reduction | stability-sensitive large runs | pure speed chasing when bf16 reduction was already sufficient |
| expert overlap | real concurrent execution exists | overlap flag is present but runtime does not overlap meaningful work |
| attention-kernel upgrades | `A`-heavy lanes with big attention share | mixed or `E`-heavy lanes where attention is a minor slice |

That table sounds modest, but it is the honest one.

## Memory shape is often the hidden governor of H200 throughput

On large accelerators it is tempting to assume memory is no longer the main issue. The receipts here say otherwise. Several of the most meaningful wins in MegaCpp come from reducing useless buffer materialization or from keeping an intermediate out of the graph entirely. The Mamba output-layer parity fix is a good example because it removes a large per-slot allocation. The DSA indexer reproducer is another because it collapses an extremely large intermediate into a much smaller fused buffer with matching math.

Why does that matter for speed instead of only feasibility? Because memory shape changes everything else. It can determine whether a target microbatch fits, whether pipeline slots remain stable, whether the runtime spends time fighting allocator pressure, and whether a lane can stay on the intended fast path instead of dropping into a fallback.

The same principle appears in the prototype-side notes around recompute and compile policy. If an explicit warmup policy or a bulky intermediate pushes the lane into instability, then the theoretical kernel speed on the steady-state path becomes irrelevant. The run never reaches the clean steady state you thought you were measuring.

For NAM56R-scale discussions this is especially important. Model family labels are not decorative. They tell you the likely pressure points. A NAM52 run and a NAM56R run can differ not just in size but in which path becomes memory-sensitive first.

## Mamba and expert paths need separate H200 narratives

Another recurring mistake is to collapse all non-dense work into one bucket. The sources point in the opposite direction. Mamba-specific work, expert-routing work, and sparse-attention work each have their own limiting factors.

The changelog describes targeted Mamba ingress fusion work and also explains why a full broader replacement is deferred. The Mamba upstream reproducer in MegaCpp then shows a subtler point: on modern Triton and H200, a seemingly obvious backward-kernel cleanup can be mostly neutral because common-subexpression elimination already removes the redundant dots. That is exactly the kind of receipt that keeps an optimization program honest. Not every attractive kernel diff translates into a big H200 win on a current toolchain.

Expert paths tell a different story. The March warmup regression report shows that regional-compile plus MoE has special sensitivity during startup. The changelog repeatedly checks router and overlap claims against runtime truth. Together these documents say that expert-heavy H200 speed is influenced as much by compile policy and routing execution reality as by pure GEMM speed.

The result is that the H200 speed story should usually be split three ways:

1. dense or mostly attention-heavy lanes
2. expert-heavy lanes with routing and regional-compile considerations
3. Mamba or recurrent-heavy lanes where projection and state behavior matter more than attention folklore

That split is far more predictive than any single accelerator-wide headline number.

## What to put in an H200 speed receipt

The right output format is boring and specific. That is good. A credible H200 speed receipt should include the model family, pattern string, compile policy, whether warmup was explicit or skipped, whether the lane is dense or MoE-heavy, any major precision or overlap knobs, and one note about dominant step share if you have it.

For example:

```yaml
family: NAM52
pattern: AEME
device: H200
lane: cuda_regional_compile
moe_enabled: true
compile_warmup: skipped
dominant_cost: expert_forward_plus_comm
attention_share: low
notes: local attention win will not move total much on this lane
```

That kind of record lets future readers compare apples to apples. It also prevents performance folklore from spreading across lanes that do not share the same anatomy.

The main conclusion is simple. H200 training speed is determined by the executed lane, not by the chip name alone. In the current stack, compile warmup policy, block mix, communication overlap, and memory shape explain more than fashionable kernel narratives do. The best wins come from removing fake work, activating the right real path, and reporting speed with enough structure that someone else can tell what was actually measured.

## References

- the public project README
- `CHANGELOG.md`
- an H200 compile warmup regression report
- a live bug audit report
- a public upstream example about Mamba linear CE
- a public upstream example about DSA indexer memory
- a public upstream example about Mamba backward vdot
