---
title: "TPU v6e Performance Deep Dive: Real MFU, Sharding Topology, and the Things That Pretended to Help"
description: "How the prototype’s TPU v6e lane actually spent time, why topology and compile amortization mattered so much, and which optimizations did not survive measurement."
date: "2026-04-18"
tags: ["tpu", "v6e", "performance", "mfu", "xla", "the POC"]
---

# TPU v6e Performance Deep Dive: Real MFU, Sharding Topology, and the Things That Pretended to Help

**TL;DR:** the important TPU v6e lesson in this stack was not a single heroic kernel win. The durable story was that topology, compile amortization, and workload shape determined whether the lane had enough useful work per chip. The docs and bug reports repeatedly show the same pattern: honest TPU performance depends on stable sharding, feasible context length, and clear separation between genuine throughput gains and “wins” caused by changed workload or broken runtime paths.

TPU performance is easy to romanticize because a good steady-state number can look dramatic. It is also easy to misread because TPU lanes make compile and sharding behavior part of end-to-end runtime in a way many GPU writeups understate. The POC’s TPU documents are valuable because they keep both truths visible.

The repo does not offer one clean headline like “v6e is fast” or “v6e is bad for hybrids.” Instead, it shows a set of constraints that interact: context length, tensor-parallel topology, compile reuse, sparse or dense attention backend choice, and whether the exact run stayed on the intended XLA path. That is a messier story, but it is the one operators can actually use.

## Topology set the ceiling before micro-optimizations did

`PHASE5_ABLATION_PLAN.md` is unusually direct about this. It records validated context ceilings on v6e hardware, notes when 256K context would require sequence parallelism or larger HBM, and ties throughput expectations to explicit TP layouts. That is exactly the right framing. Before asking whether some local optimization helped, you need to know whether the topology left enough real work per chip.

A topology that technically fits but starves each chip of useful batch or sequence work will always look worse than a cleaner layout with fewer coordination costs. That is not a moral failure of the accelerator. It is the expected result of asking the topology to carry a workload it does not host efficiently.

| Condition | What usually happens on v6e |
| --- | --- |
| Enough work per chip, stable compile reuse | throughput can look healthy and repeatable |
| Memory-feasible but communication-heavy topology | MFU drops and coordination dominates |
| Long context beyond validated range without SP | run becomes topology- or memory-limited before kernels matter |
| Unstable compile or fallback path | reported throughput stops reflecting the intended lane |

That table sounds obvious, but it is the part many postmortems skip. They jump from number to number without first naming the topology regime that produced each number.

## Compile amortization is part of TPU performance, not an afterthought

The TPU-specific docs and bug reviews repeatedly reinforce this point. A lane that recompiles too often, invalidates caches, or falls onto unstable shapes is not operationally fast even if one hot loop is efficient after compilation. On TPU, compile behavior is part of performance.

This is one reason the XLA-focused bugs matter so much to performance interpretation. The strict bug pass includes failures around reduction paths, static-grad materialization, and fallback behavior. Those may read like correctness or robustness notes, but they are also performance notes. If a lane silently changes execution strategy or loses shape guarantees, any measured throughput becomes harder to trust.

The lesson is straightforward: before celebrating MFU, verify that the lane is on a stable compiled path and that cache reuse is doing what you think it is doing. Otherwise you are benchmarking a moving target.

## Workload shape matters more than slogan-level comparisons

The TPU lane in this project spans dense, sparse, and hybrid experiments. `SOTA_COMPARISON.md` mentions TPU-native and XLA-safe variants for several ideas, including sparse attention, threshold-based depth routing, and XLA-safe implementations of techniques that originated elsewhere. That is useful context because it explains why not all TPU results should be collapsed into one “the TPU lane.”

A shallow dense run, a hybrid NAM-style run, and a sparse-attention experiment can all be perfectly valid TPU workloads while stressing completely different parts of the system. Comparing their tokens-per-second as if they were interchangeable workloads is a category error.

The hybrid pattern notation helps here just as it does on GPU. If a run uses more `E` positions, more recurrent behavior, or a different long-context plan, that changes the meaning of the performance result. Pattern-aware reporting is therefore more honest than generic throughput bragging.

## Long context was a first-order systems constraint

`PHASE5_ABLATION_PLAN.md` is also strong on long-context limits. It notes that TP=8 validated 128K context on the target hardware, while 256K would require sequence parallelism or larger hardware. That single note explains a lot about the lane’s real boundaries.

Long-context ambition on TPU is not just a kernel problem. It is a memory-shape and topology problem. If the sequence length forces a layout that cuts effective work per chip too aggressively, utilization drops even if the math kernels themselves are competent. That is why long-context planning belongs in the same conversation as MFU.

It also explains why some optimizations “pretended to help.” A tweak that improves a smaller or shallower workload may not survive when the real target is a long-context hybrid run. The workload shape changed, so the bottleneck moved.

## Sparse and XLA-safe paths changed what was measurable

The prototype TPU work is notable for trying XLA-safe or TPU-native implementations rather than assuming CUDA-first logic will port cleanly. `SOTA_COMPARISON.md` calls out threshold-based depth routing on TPU, clustered sparse attention as experimental, and XLA-safe substitutions where the original method depended on GPU-only behavior.

That matters because TPU performance should not be narrated as “the same algorithm, different hardware” when the runtime path is genuinely different. If a TPU-safe path avoids gather/scatter or uses a different masking strategy, the performance result is describing a different systems contract, not just a different device.

This is not a problem. It is normal engineering. But it means the report should say so clearly. Otherwise the reader cannot tell whether the gain came from hardware, topology, algorithmic substitution, or all three.

## What did not survive measurement

A good deep dive should also say what did not matter enough. The stable lesson from the docs is that second-order tweaks rarely outranked topology and compile reuse. If per-chip work was too low, or compile amortization was poor, many smaller wins were simply not large enough to change the operator-facing story.

That does not make those smaller optimizations worthless. It just means they were not the main plot. The main plot was whether the lane fit into a stable, reusable, memory-feasible topology with enough real work per chip.

This is the part that “pretended to help.” A local speedup can look exciting in isolation and still fail to move the full training lane once topology and compile dominate again.

## A grounded TPU launch shape

The ablation-plan examples show a recurring TPU launch style: explicit tensor-parallel choice, explicit sequence length, explicit total batch target, and TPU-safe backend flags. A representative shape looks like this:

```text
A representative TPU benchmark launch pinned total batch size, tensor parallel degree, the current kernel path, compile mode, and the XLA flash-attention switch in one reproducible command.
```

The exact launcher is less important than the structure. TPU results in this stack are always tied to a topology and backend story. If either story changes, the number has to be reinterpreted.

## How to read a TPU performance result correctly

A useful TPU v6e performance note should answer five questions.

1. What topology was used?
2. Was compile amortized or was this effectively a cold-start measurement?
3. What sequence length and workload family were active?
4. Did the run stay on the intended XLA-safe path without silent fallback?
5. Did the gain survive when moved back to the target workload shape?

If those answers are missing, the result is hard to operationalize. If they are present, even a modest gain can be more valuable because you know what it means.

## The durable lesson

The durable lesson from the TPU v6e work is not “optimize harder.” It is “measure the right regime.” Stable throughput on TPU comes from pairing the right workload with the right topology and compile behavior. Long-context ambitions have to respect validated memory and sharding limits. XLA-safe paths need to be verified as the active paths. And small wins should be treated skeptically until they survive the real workload.

That is a better standard than chasing one flattering MFU screenshot. It is also exactly what the repo materials encourage. They keep topology, compile reuse, sequence length, and backend choice in the same frame. That is what makes the results believable.

## Why validated ceilings are more valuable than isolated peaks

One of the strongest habits in the TPU planning documents is the emphasis on validated ceilings. Knowing that a particular context length or topology is repeatably feasible is often more operationally valuable than a single faster run at a friendlier shape. That may sound conservative, but it is the right bias for a lane where compile and sharding behavior can shift the regime so easily.

A validated ceiling answers a planning question: what is the largest shape we can rely on? An isolated peak often answers only a marketing question: what is the nicest number we saw once? For TPU work, the planning question is usually the more important one because downstream recipe choices depend on it. Context expansion, sparse-attention experiments, and hybrid schedules all need a believable operating envelope.

This is why the context-limit notes in `PHASE5_ABLATION_PLAN.md` are more important than they might first appear. They turn TPU performance from an anecdote into a scheduling input.

## The honest TPU report is specific about what changed

A good v6e report therefore names the real source of improvement. Was it a better TP layout? More compile reuse? A workload with less costly family composition? An XLA-safe path that avoided expensive gather or scatter? The repo materials repeatedly suggest that this specificity is the difference between useful evidence and noise.

That specificity also helps prevent over-generalization. A result that is excellent for a shorter-context, cleaner-shape lane may still fail to carry over to the exact NAM-style hybrid workload you eventually care about. The report should say so directly. That is not weakness. It is how teams avoid spending the next week trying to port a win that never actually targeted the production regime.

In that sense, the TPU deep dive is not merely about performance. It is about evidence quality. The better the report names topology, compile state, and workload shape, the more likely the next optimization step will start from reality rather than wishful thinking.

## References

- `PHASE5_ABLATION_PLAN.md`
- `SOTA_COMPARISON.md`
- `current_live_bugs_2026-03-07_strict_pass3.md`
- the main model runtime module
- `eval_doc.ru.md`
