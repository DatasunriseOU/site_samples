---
title: "Profiler-Guided Optimization: Start With the Runtime Story, Not the Theory"
date: 2026-04-18
author: MegaCpp Engineering
tags: [profiling, optimization, nsys, performance, distributed-training]
summary: >
  The most useful performance work in the MegaCpp stack came from following real
  runtime receipts: compile warmup behavior, elementwise overhead, layout
  contracts, and family-specific hot paths. The pattern is simple: profile the
  actual lane, isolate the dominant cost, then change only what the receipt
  justifies.
description: >
  A grounded guide to profiler-led optimization using the reports, code
  comments, and configuration surfaces in the MegaCpp repos.
---

# Profiler-Guided Optimization: Start With the Runtime Story, Not the Theory

The fastest way to waste engineering time is to optimize the thing that merely sounds expensive. The useful workflow is to profile the real lane, identify the dominating cost center, and then make the narrowest possible change that matches the measured story.

Performance engineering gets worse when it becomes ideological. One team decides fused kernels are always the answer. Another decides topology dominates everything. A third blames the compiler by default. Real systems are not that tidy. The same stack can be limited by attention kernels in one lane, compile warmup in another, and expert dispatch semantics in a third.

MegaCpp's reports and comments are useful because they refuse to compress those cases into one narrative. Instead of declaring a universal bottleneck, they preserve a trail of receipts tied to exact runtime conditions. That is what makes profiler-guided optimization practical rather than ornamental.

## What the code already tells you to measure

Good profiling starts before the profiler runs. The source tree already contains hints about where large costs live.

In the main model runtime module, the config comments call out major recompute wins by family: recomputing expert GEMM in backward saves substantial memory across many `EBlock` instances, M2RNN recurrence has its own recompute surface, and Mamba convolution has another. The same file also notes a target around reducing elementwise-add overhead, referencing a previously observed high share in `nsys` profiles. Those comments are not proofs, but they are strong priors.

The public performance reports strengthen them. One H200 compile-warmup regression note shows a case where the dominant problem was not an individual kernel at all. The lane stalled in explicit compile warmup, and the fix was to skip warmup for a narrow `regional_compile + MoE` case. That is a textbook example of profiler-guided thinking: do not attack the fancy kernel if the system is blocked before the first real step.

Likewise, the strict live-bugs report records XLA-specific optimizer and reduction problems that would distort any performance interpretation if left unresolved. A lane with shape instability or silent fallback is not ready for micro-optimization. The profiler can only tell the truth if the execution path is already honest.

## Start with the dominant phase, not the hottest symbol

One of the easiest mistakes in profiling is to focus on the hottest symbol in a trace without first asking which phase dominates end-to-end runtime. Cold-start compile, steady-state forward, backward, optimizer step, and communication overlap are different phases. The bottleneck can move across them.

The H200 warmup report is the cleanest illustration. If warmup never completes, then no downstream kernel-level theory matters. The correct optimization was policy-level: disable explicit warmup for the affected lane and let lazy compile plus cache growth take over. It was narrow, evidence-backed, and higher leverage than speculative kernel surgery.

That is a durable lesson. Before changing math kernels, ask:

| Question | Why it matters |
| --- | --- |
| Is cold start or compile dominating? | A compile-bound lane needs policy and cache work first |
| Is the steady-state bottleneck compute, memory, or exchange? | Different classes of fixes follow |
| Is the runtime even on the intended path? | Fallbacks and disabled fusions invalidate conclusions |
| Is the hot path family-specific? | `ABlock`, `EBlock`, `MBlock`, and `RBlock` stress different subsystems |

This phase-first view prevents a common failure mode: spending days shaving a few percent from steady-state compute in a lane that is mostly paying startup tax or dispatch overhead.

## Hybrid models demand family-aware profiling

MegaCpp is not a homogeneous transformer stack. It separates `ABlock`, `MBlock`, `EBlock`, and `RBlock`, and it uses pattern strings such as NAM52 or `AEMEAEMEAEMR` to define mixed schedules. That means the profiler story is rarely “the model is slow.” It is more often “one family inside the model is dominating under this topology and this configuration.”

That distinction matters because the optimization levers differ.

`ABlock` issues tend to be about attention kernels, latent-cache layout, normalization fusion, or sequence-parallel interactions. `EBlock` issues often involve router semantics, expert dispatch, shared-expert overlap, or compile behavior under MoE. `MBlock` and `RBlock` bring stateful sequence-mixing behavior that can shift the memory and recompute balance.

If you ignore those distinctions, you get misleading conclusions. A trace dominated by expert exchange in an `E`-heavy region does not prove attention is solved. A clean attention profile in a shallow dense lane does not prove a NAM56R hybrid recipe is healthy. Family-aware profiling is not a luxury; it is required by the architecture.

## Narrow changes beat broad rewrites

The strongest optimization examples in this stack are narrow. The warmup regression fix did not rewrite the compiler interface. It changed the default policy for a specific CUDA lane shape. The comments around recompute surfaces do not claim global checkpointing magic. They identify concrete families where backward recompute buys real memory back.

That pattern should guide performance work:

1. Capture the lane that matters.
2. Name the dominant cost with evidence.
3. Change the smallest surface that could plausibly fix that cost.
4. Re-profile the same lane.

This approach matters even more in a fast-moving training stack. Broad rewrites produce too many moving parts at once. If throughput changes, you no longer know why. Narrow changes preserve causality.

The same principle applies to layout work. If a trace points to elementwise overhead near block boundaries, the right next step may be fusion or boundary simplification in that exact path, not a repo-wide rewrite of all normalization behavior.

## Use profiler output to reject fake wins

Another benefit of receipt-driven optimization is that it helps reject wins that are not real. The repos contain repeated warnings against fake or misleading throughput numbers. MegaCpp’s NAM56R recipe explicitly warns against mock-data benchmarking because it produces throughput that does not match real training. That is more than a benchmarking footnote. It is an optimization rule: if the workload is fake, the speedup may be fake too.

The same logic applies to compile amortization, attention shortcuts, and reduced-workload comparisons. A faster run is only meaningful if the reader can tell whether the same work was performed. In hybrid stacks, this is especially important because a run with fewer active `EBlock` stresses, simpler routing, or different sequence length can look like a systems win when it is really a workload change.

A compact verification checklist helps:

```text
same pattern?
same routed-expert settings?
same sequence length?
same compile policy?
same substrate and parallel topology?
same real-data path?
```

If the answer is no on multiple lines, the speedup claim needs careful qualification.

## Profiling MoE lanes means profiling the whole contract

MoE performance is a good example of why “profile the kernel” is not enough. In MegaCpp, MoE carries routing mode, score function, top-k, shared experts, grouped routing, optional FP8 experts, and fused execution choices. Any of those can change the exchange and compute profile.

That is why a serious MoE optimization pass should treat the following as part of one system:

- router math and top-k selection,
- token permutation and dispatch,
- expert compute,
- shared-expert overlap,
- aux or z-loss side work,
- compile behavior of the exact lane.

The warmup report again shows how easy it is to misdiagnose this. A MoE lane that never escapes compile warmup is not revealing its expert-kernel bottleneck yet. Fix warmup first, then profile the real step path.

## When topology outranks kernel work

Not every performance result points to code-level optimization. Sometimes the measured bottleneck is a topology or workload-shape issue. TPU bring-up notes repeatedly emphasize memory-feasible topology, validated context limits, and the importance of enough per-chip work. That is an important complement to GPU-focused profiling.

Profiler-guided optimization therefore includes the discipline to say “this is not a kernel problem.” If communication shape, pipeline imbalance, or compile amortization dominates, then topology changes or schedule changes may be the highest-value optimization. That answer can feel less glamorous than a fused kernel, but it is often more honest.

## A representative optimization workflow

The following pattern matches what the repos reward in practice:

```text
example profiler receipt run:
  profiler = nsys
  traces = cuda,nvtx,osrt
  sampling = none
  output = run_receipt
  workload = hybrid training lane with AEMEAEMEAEMR and MoE enabled
```

The command shape varies, but the point is constant: capture the real lane, preserve the receipt, and compare before and after on the same workload. Then connect the result back to the code surface you actually changed.

That connection matters. A profiler artifact without a code-local explanation is hard to trust. A code change without a matching receipt is easy to overclaim. The stack works best when those two pieces stay paired.

## What should survive after the exact numbers move

Even when the exact traces change, a few rules remain stable.

First, optimize phases in order: startup policy, steady-state hot path, then second-order cleanup. Second, profile hybrid models by layer family, not as a monolith. Third, reject speedups that rely on fake data or changed workload shape. Fourth, prefer narrow changes with clear causal stories. Fifth, keep receipts. A report that says “this removed the warmup stall” or “this reduced add-bound overhead at block boundaries” is more durable than one that only posts a good percentage.

That is what profiler-guided optimization means in practice. It means letting the runtime story tell you what to do next, rather than forcing the runtime to validate a theory you already wanted to believe.

## Receipts beat intuition when compile and runtime interact

One reason this workflow matters so much in MegaCpp is that compile policy and runtime behavior are entangled. A lane may appear to have a kernel problem when the real issue is that the compiler was asked to warm up a toxic path, or that a supposedly steady-state trace is still dominated by compile-time side effects. That is why compile-warmup receipts need to be separated from steady-state throughput receipts.

This is also why keeping a historical trail matters. If a system previously needed an escape hatch such as no-warmup policy for a specific class of runs, then any later re-enable should be treated as a new hypothesis, not as a permanent truth. The report shows what happens when that caution is ignored: the runtime regresses back into warmup trouble and the profiler story becomes distorted until the policy is narrowed again.

A disciplined optimization culture therefore does two things at once. It measures the current lane, and it preserves enough receipts to explain why the lane is different from older ones. Without that context, teams keep re-learning the same performance lessons.

## Family-specific profiling produces better optimization queues

Because the model is hybrid, a single optimization queue is rarely the right queue. The useful queue is often segmented by family. Attention-facing work may target projection fusion or layout cleanup. Expert-facing work may target routing stability, dispatch backend choice, or overlap. Recurrent work may target state materialization or recurrence checkpointing.

This segmented view has a practical benefit: it prevents the performance backlog from becoming a pile of generic “speed up training” tasks. Instead, each queue stays tied to a code surface and a measurable symptom. That is exactly how the comments and reports in these repos read. They point to concrete files, concrete lane shapes, and concrete receipts.

When teams skip that discipline, they often produce patches that are impossible to evaluate cleanly. The change touches multiple families, the benchmark shape drifts, and the resulting speedup cannot be trusted. Profiler-guided optimization is partly about tools, but it is also about change hygiene.

## References

- the main model runtime module
- Public H200 compile-warmup and live-bug review notes in the MegaCpp repo
- Public NAM56R recipe and ablation documentation
