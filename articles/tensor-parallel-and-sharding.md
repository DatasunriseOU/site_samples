---
title: "Tensor Parallel and Sharding: What Actually Splits, What Still Stays Global"
date: 2026-04-18
author: MegaCpp Engineering
tags: [tensor-parallel, sharding, distributed-training, sequence-parallel]
summary: >
  Tensor parallelism is a precise matrix-partitioning contract, not a general
  answer to every memory problem. It coexists with expert,
  sequence, context, and pipeline strategies, each solving a different resource
  constraint.
description: >
  A code- and doc-grounded walkthrough of tensor parallelism in public hybrid
  recipes, including where TP helps, where it does not, and how it fits into
  hybrid NAM52 and NAM56R workloads.
---

# Tensor Parallel and Sharding: What Actually Splits, What Still Stays Global

Tensor parallelism works best when you treat it as a narrow contract around matrix dimensions and communication points. It reduces per-rank parameter and activation pressure for the surfaces it shards, but it does not automatically solve expert routing, latent-cache residency, recurrent-state costs, or pipeline imbalance. The contract is visible in [this Mamba3 TP partition-size sample](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/MegaCpp source repository/megatron/tensor-parallel-and-sharding__mamba3_tp_partition_sizes__v1.py).

## Code and notes

- [Mamba3 TP partition-size sample](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/MegaCpp source repository/megatron/tensor-parallel-and-sharding__mamba3_tp_partition_sizes__v1.py)
- [PyTorch DTensor docs](https://pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch tensor-parallel docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)

Distributed-training explanations often make TP sound broader than it is. The phrase “split the model across GPUs” is technically true but hides the operational detail that only certain tensors and computations are partitioned. The rest of the system still has to cooperate.

The public NAM56R-style notes in this repo are a good corrective. They keep TP claims tied to sequence length, expert ownership, and hybrid block families instead of presenting TP as a universal default. TP is a tradeoff tool whose value depends on the layer family, the backend, and the features you need to preserve.

## What TP actually promises

At its core, tensor parallelism splits compatible linear-algebra surfaces across ranks. In practice that means large projections, feed-forward layers, and other tensor-heavy operations can be partitioned so no single rank owns the full weight or full intermediate activation for that surface. The payoff is lower per-rank memory and, sometimes, better aggregate throughput when the communication pattern is healthy.

The public code excerpts make this concrete. Dense tensor partition size is one contract; expert ownership is another. On CUDA, TP usually maps to explicit process-group collectives. On XLA, the same idea is expressed through mesh sharding. That separation already tells you TP is substrate-specific in implementation even when the conceptual goal is the same.

| Parallel mode | Primary target | What it does not automatically solve |
| --- | --- | --- |
| TP | dense tensor math and large projections | expert routing, pipeline loss plumbing, recurrent-state semantics |
| EP | expert ownership and expert compute distribution | dense attention or Mamba projection pressure |
| SP | sequence-dimension activation pressure | expert ownership or topology imbalance |
| CP | long-context partitioning | local per-token math costs |
| PP | stage-level model residency | fine-grained tensor splits inside a stage |

Keeping these roles separate prevents a lot of confusion. If you ask TP to solve a problem owned by EP or CP, you will either be disappointed or over-engineer the wrong surface.

## Why TP matters in hybrid architectures

The hybrid layout notes make hybrid architectures explicit with `ABlock`, `MBlock`, `EBlock`, and `RBlock`. That matters because TP interacts differently with different families.

For attention-bearing layers, TP often targets the large QKV and output projections. For dense FFN paths, it targets the usual expansion and contraction matrices. But for `EBlock`, which may be MoE-based, there is a second sharding story: expert ownership and expert tensor degree. The runtime configuration explicitly distinguishes `tp_degree` from `expert_tp_degree` so the model can run configurations where dense math and expert math are partitioned differently.

This distinction is more than hygiene. It is what lets a hybrid model say, in effect, “attention is tensor-parallel, experts follow a different ownership plan.” Without that separation, MoE and dense paths would interfere with each other’s layout assumptions.

NAM56R makes this concrete. The recipe uses `AEMEAEMEAEMR`, so the model alternates family types with different communication and memory behavior. A single “the model is TP=2” label does not fully describe what is happening. You also need to know which `E` positions have their own expert partitioning story and which `M` or `R` positions preserve custom mixer behavior that may argue for a different global mode.

## Sequence parallel is not optional hand-waving

The related public notes in this repo are particularly useful here because they show where TP runs out of room. Long-context goals still require sequence parallelism or context parallelism, and larger context windows are still constrained by HBM and topology even when TP is already active. That is exactly the kind of grounded caveat missing from shallow TP explanations.

TP reduces pressure along one dimension. Long sequences can still explode activation residency along another. If your bottleneck is context length, then SP or CP may be the right next tool, not more TP.

That is why an honest distributed story for this stack sounds layered, not monolithic:

- use TP for tensor-heavy projections,
- use SP when sequence activations dominate,
- use EP when expert banks dominate,
- use PP when full-model residency exceeds one device,
- use CP when the context window itself becomes the limiting axis.

The important part is not the list. The important part is admitting that each item answers a different resource question.

## TP is a feature tradeoff too

The public writeups here do something many performance guides avoid: they tie parallel choice to feature availability. A layout that looks optimal for dense projections may still be wrong for MoE ownership, Mamba-style mixers, or recurrent memory blocks.

This is an important operational truth. Parallelism choice is not only about memory and throughput. It can also constrain which kernels, mixers, or scheduling tricks are available. A theoretically better TP layout is not automatically the right choice if it forces you off the feature set you actually need.

This is especially relevant in hybrid models where `RBlock` and `MBlock` may represent custom sequence-mixing experiments. The right question is not “can TP be turned on?” It is “what exactly remains intact when TP is turned on?”

## The XLA and TPU wrinkle

MegaCpp also makes clear that TPU and XLA lanes should not be narrated as if they were CUDA with different silicon. The docs around TPU, long context, and sparse attention repeatedly talk in terms of XLA-safe paths, sharding, and validated topologies. That matters because the practical behavior of TP-like sharding on XLA depends heavily on compile stability and sharding annotations.

In other words, a CUDA TP story often emphasizes collectives and overlap. An XLA sharding story often emphasizes whether the compiler preserves the intended partition and whether the run remains shape-stable enough to amortize compile cost. Those are related but not identical concerns.

That is why cross-substrate comparisons need careful language. Saying “TP helped” is too vague. You need to say whether the improvement came from lower per-rank weight residency, cleaner sequence partitioning, better compile shape, or a different topology that finally made the workload feasible.

## A representative configuration surface

A realistic TP-bearing configuration in this stack includes more than one flag because TP lives inside a larger sharding plan.

```yaml
mode: nemo_native
pattern: AEMEAEMEAEMR
tensor_parallel: 2
sequence_parallel: true
expert_tensor_parallel: 1
context_parallel: 1
pipeline_parallel: 1
```

The exact names vary across launchers, but the structure is the point. TP is one dimension in a layout tuple. It is not the whole tuple.

This also explains why benchmarking can go wrong. If you compare two runs where TP changed but sequence parallel, compile policy, or expert partitioning also changed, you do not yet know what caused the result. A disciplined comparison isolates one axis at a time.

## What TP cannot rescue

It is worth being explicit about the limits because those limits are what motivate the rest of the stack.

TP does not make router aux losses disappear. It does not fix MoE dispatch imbalance. It does not make pipeline-stage loss reconstruction automatic. It does not erase compile warmup problems. It does not eliminate the family-specific state costs of `MBlock` or `RBlock`. And it does not guarantee long-context feasibility by itself.

That sounds obvious when stated plainly, but many distributed writeups smuggle those expectations in indirectly. They report one good TP result, then let the reader infer a broader cure. MegaCpp's docs and recipes are better because they keep naming the other axes.

## The durable takeaway

The durable takeaway is simple: tensor parallelism is a powerful but narrow tool. Use it where dense tensor math is the limiting factor. Combine it with SP, CP, EP, or PP when a different axis is limiting. Preserve architecture awareness in hybrid models like NAM52 and NAM56R. And always verify what changed besides TP before claiming a win.

That is how the MegaCpp stack treats the problem. The recipes define mode tradeoffs explicitly. The config separates TP from expert tensor degree. The docs call out when sequence parallel or larger-topology choices are required. And the hybrid architecture keeps family-specific behavior visible instead of flattening everything into one anonymous layer stack.

That is a much better foundation than “TP splits the model.” It tells you what split occurred, why it mattered, and what still remained global afterward.

## TP interacts with architecture math, not just deployment scale

The NAM56R recipe is also useful because it encodes concrete architectural math alongside the parallel modes. Hidden size, head count, kv-head count, sequence length, and FFN sizes are all fixed in one place. That makes it easier to reason about why TP helps a given lane. It is not helping an abstract “big model.” It is helping specific projection shapes.

For example, the recipe records 56 attention heads with 8 kv-heads and a hidden size of 3584. Those numbers imply concrete tensor shapes for QKV and output projections. When TP is enabled, those are the kinds of surfaces being partitioned. That is much more informative than saying the model is “too large for one device.” It tells you what algebra is actually being split.

The same reasoning explains why TP does not magically solve every hybrid cost. `RBlock` and `MBlock` can still carry state and recurrence costs that are not reducible to the same projection split story. `EBlock` may have its own expert partitioning plan. And long-context pressure may still be dominated by sequence-dimension activation size rather than by weight residency.

## Why comparisons need one-axis-at-a-time discipline

The TPU planning docs quietly teach another important TP lesson: context, topology, and TP often move together, which makes careless comparisons misleading. A run with higher TP degree may also have a different feasible batch, a different sequence limit, or a different communication shape. If you celebrate the new throughput number without isolating those changes, you do not yet know whether TP was the main cause.

The right way to compare TP changes is boring but necessary. Hold the architecture pattern constant. Hold the routed-expert settings constant if MoE is present. Hold the sequence length constant unless you are explicitly studying long-context feasibility. Then vary TP and observe what changed in memory headroom, compile behavior, and steady-state throughput.

That discipline is what turns TP from a folklore knob into an engineering tool. It also makes it much easier to explain why a seemingly good TP configuration was rejected. Sometimes the answer is that it interfered with the required feature path. Sometimes it solved one resource problem and worsened another. The important part is that the reason stays tied to the measured lane.

## References

- [Mamba3 TP partition-size sample](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/MegaCpp source repository/megatron/tensor-parallel-and-sharding__mamba3_tp_partition_sizes__v1.py)
- [Hybrid layout notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md)
- [Distributed debugging notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md)
- [PyTorch DTensor docs](https://pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch tensor-parallel docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Megatron Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron Core context parallel guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron Core MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
