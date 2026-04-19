---
title: "What Megatron Can and Cannot Split"
date: 2026-04-18
author: Engineering Team
tags: [megatron, tensor-parallel, pipeline-parallel, moe, mamba, nam56r]
summary: >
  Megatron-style partitioning is powerful, but not universal. Modern runtimes make the
  real boundary clear: dense regular computation splits cleanly, heterogeneous
  ownership often does not, and some surfaces must remain explicit.
description: >
  A grounded look at split-friendly and split-hostile model surfaces: TP, SP,
  PP, EP, recurrent state, side embeddings, and why some boundaries remain
  architectural rather than automatic.
---

Megatron is excellent at partitioning regular dense computation. It is much less magical on heterogeneous ownership surfaces such as expert routing, recurrent state transitions, side-channel embeddings, and topology-sensitive pipeline boundaries. The practical skill is not maximizing how much gets split, but deciding which surfaces should be split aggressively and which should remain explicit architectural boundaries.

Megatron-style systems are compelling because they offer a powerful vocabulary for scale: tensor parallelism splits weight matrices, sequence parallelism changes activation ownership on the TP axis, context parallelism changes long-sequence ownership across ranks, pipeline parallelism partitions depth, and expert parallelism distributes expert banks. On slides that can look close to universal. In code it is not universal at all.

The right question is therefore not "can Megatron split the model?" The right question is "which surfaces preserve their meaning when split?" The answer is visible in current runtimes because the stack supports TP, SP, PP, optional virtual staging, expert-distribution modes, and hybrid family layouts. Reading those code paths makes one thing clear: regular dense math is naturally split-friendly, but control-heavy or topology-bearing surfaces still need explicit ownership rules.

## What Megatron splits very well

The strongest targets are the ones the framework was effectively designed around: repeated dense projections and repeated depth. Attention projections, dense MLP projections, and the inner body of standard blocks all have predictable shapes and communication patterns. Those properties are exactly what tensor and sequence parallelism want.

That is why TP and SP remain the most transferable parts of the Megatron story. They work best when the runtime can treat the computation as regular linear algebra with known collective boundaries. CP solves a different problem: not dense matrix ownership, but long-context sequence ownership. The dense path does not become trivial, but it stays legible enough for compiler transforms, sharding strategies, and checkpoint rules to agree.

Public distributed-training notes reflect that orientation. They are full of explicit decomposition logic, but the places where decomposition is cleanest are the places where shapes and ownership are stable. Likewise, the training configuration exposes TP, SP, PP, virtual-stage, and expert-distribution flags because these are not theoretical axes. They are the real knobs the runtime can use when the underlying structure is cooperative.

| Surface | Split quality | Why it works |
| --- | --- | --- |
| Dense attention projections | Excellent | predictable shapes and collectives |
| Dense MLP projections | Excellent | same regularity as attention projections |
| Repeated layer depth | Strong | PP and virtual staging map well when boundaries are meaningful |
| Embedding and output head edges | Good, but explicit | works once edge ownership is fixed |

That is the part people usually mean when they say Megatron scales well. They are not wrong. They are just describing the regular half of the system.

## Pipeline splitting still depends on architectural topology

Pipeline parallelism is often described as "split the layers by depth." That description is incomplete. Real pipeline splitting is only valid after the system decides what each stage owns besides the obvious repeated block range.

Modern runtimes make this explicit. The runtime has to reason not only about how many layers land on each stage, but also about which stage owns embeddings, heads, RoPE-related state, auxiliary embeddings, and family-specific support modules. That means PP is not merely arithmetic partitioning. It is architectural partitioning.

This is one reason pattern notation matters. A family like NAM56R is not just a depth number. `AEMEAEMEAEMR` implies heterogeneous block roles across the stack. If you cut depth carelessly, you may preserve layer count while damaging ownership semantics. A stage boundary that is fine for dense `ablock` repetition may be awkward if it slices through an `eblock` routing-heavy region or an `rblock` state-carrying region.

The practical lesson is simple: PP works best when the runtime exposes the topology rather than hiding it. Weighted partitioning, explicit stage boundaries, and family-aware placement are signs of maturity, not of failure.

```text
Dense repeated body   -> split by regular depth rules
Family-specific edges -> assign explicitly
Topology-heavy seams  -> make visible instead of pretending they are generic
```

That block describes what the current stack is already doing in spirit, even when the exact implementation details keep evolving.

## Expert weights split cleanly; expert routing does not

MoE is the clearest example of the difference between split-friendly math and split-hostile control flow. Expert parallelism is excellent at distributing expert banks. Once the heavy feed-forward weights are organized per expert, sharding them across workers is natural. That is the good part.

The hard part is routing. Token assignment to experts is data dependent. Load is uneven. Communication often requires all-to-all behavior or carefully managed dispatch/combination steps. None of that disappears just because expert weights were partitioned successfully.

The training configuration exposes several MoE-related choices precisely because the runtime has to care about these distinctions. The stack needs to know not only whether MoE is enabled, but also how expert data parallelism and expert tensor parallelism are configured, whether specific distribution or packing modes are legal, and how those choices interact with checkpointing and recomputation. Those are routing-and-ownership concerns, not just GEMM concerns.

| MoE surface | Split status | Real limitation |
| --- | --- | --- |
| Expert weights | Very splittable | natural EP target |
| Expert compute kernels | Often highly optimizable | still governed by kernel contracts |
| Token routing | Only partially abstractable | live token distribution drives communication |
| Zero-token expert handling | Must stay explicit | autograd and reduction assumptions can break |

This is why "Megatron solves MoE scaling" needs to be read narrowly. It solves a major fraction of the expert-weight problem. It does not erase routing as a systems problem.

## Recurrent and Mamba-style blocks resist the usual split intuition

Hybrid families complicate the picture further because not every block is attention-shaped. NAM56R keeps `A`, `M`, `E`, and `R` roles visible for a reason. The `M` and `R` surfaces carry different semantics from a standard attention block, and those semantics matter when deciding what can be partitioned mechanically.

A recurrent transition cares about state evolution across steps. A Mamba-style mixer may have fused and split execution paths, internal convolution/scan structure, and different performance sensitivities from attention. Those are not reasons to avoid splitting altogether. They are reasons to avoid pretending that the same split rule applies everywhere.

This is where many architecture discussions go wrong. They ask whether the framework can split the model, when the real question is whether the state semantics survive the split. Sometimes they do. Sometimes the clean answer is to split the dense projections around the block while leaving the state-bearing contract explicit.

That is not a weakness of the framework. It is an admission that architecture still exists.

## Side-channel embeddings and auxiliary inputs are ownership problems first

Another class of surfaces that resists generic partitioning is side-channel input. The stack contains optional structure-aware embedding paths and other non-default input enrichments. Those are not naturally split-first abstractions. They are representational features whose placement has to be designed.

If an auxiliary embedding should exist only on the first stage, or should remain visible after a stage reshuffle, or has to preserve checkpoint compatibility when the split geometry changes, then the problem is no longer only about tensor shapes. It is about ownership.

This is why it is misleading to ask whether Megatron can split "the whole model." It can split many compute surfaces inside the model. It cannot automatically invent the ownership rules for every auxiliary channel. Someone still has to define where those modules live, who materializes them, and how checkpoint state is interpreted.

## Checkpointing and recompute expose the same boundary

One of the most useful signs of the true split boundary is that checkpoint and recompute logic tend to fail in the same places that naive partitioning fails. Regular dense regions are easier to shard, checkpoint, and replay. Irregular ownership regions accumulate edge cases.

The training configuration is valuable here because it makes many of those interactions explicit: options around pipeline shape, virtual stages, MoE distribution, compile behavior, and related runtime toggles all affect what can be recomputed cleanly and what must remain visible to the runtime. A system does not need that many knobs if everything is truly generic.

That does not mean the design is bad. It means the design is honest. Once the model combines dense math, expert routing, recurrent state, and side inputs, the runtime needs declared boundaries.

| Category | Usually safe to split aggressively? | Usually needs extra ownership logic? |
| --- | --- | --- |
| Dense projections | Yes | rarely |
| Repeated depth body | Yes, with stage planning | sometimes |
| Expert banks | Yes | yes, around routing |
| Long-context token ownership | Yes, when the model is CP-aware | yes, around sequence exchange and gathers |
| Recurrent/stateful surfaces | Sometimes | often |
| Side-channel embeddings | Sometimes | usually |

The recurring pattern is obvious: regular compute loves splitting, heterogeneous ownership does not.

## The best systems do not maximize splitting; they maximize useful splitting

The strongest design lesson from the repo is that the right goal is not universal automatic partitioning. The right goal is useful splitting plus visible exceptions. Dense math should be split aggressively. Expert weights should be split aggressively. Routing, state transitions, and edge ownership should remain explicit where necessary.

That is a more mature target than trying to force every surface into a generic decomposition rule. It is also more maintainable. Engineers can debug a system where exceptions are visible and justified. They struggle in systems that claim everything is generic while quietly encoding special cases everywhere.

The same point also explains why hybrid-family notation is worth preserving. If a report mentions NAM52 or NAM56R, that is not branding. It is a reminder that the stack contains different block families with different split behavior. A pattern string like `AEMEAEMEAEMR` is useful because it signals where automatic assumptions stop being safe.

## What Megatron does not split for you

Megatron can partition dense tensors, pipeline stages, expert groups, and long-context work when the ownership model is explicit. It does not erase the need to define architectural boundaries.

Distributed checkpointing, often shortened to DCP, is not a compute split axis. It governs how state is saved and restored.

Activation recompute and checkpointing are also not partitioning modes. They change the memory and compute tradeoff, but they do not decide where parameters, tokens, or experts live.

Two taxonomy corrections matter in practice. First, SP and CP are not interchangeable token-splitting names. SP is a TP companion for activation layout; CP is a long-context ownership strategy. Second, EP is not a substitute for TP or PP. EP distributes expert banks and routed-token transport; TP and PP still own dense math and depth placement.

The remaining hard cases are the ones where ownership is conditional or family-specific: routed control flow, auxiliary losses, cross-stage side channels, recurrent state handoff, and expert-local bookkeeping. Those still need explicit contracts even when the dense path is fully under Megatron control.

## So what can Megatron split, and what can it not?

Megatron can split regular dense computation extremely well. It can split expert banks well. It can split depth well once stage boundaries are chosen meaningfully. It cannot automatically dissolve routing complexity, recurrent state semantics, or auxiliary-input ownership into linear algebra.

That is the boundary that matters in practice. Once you accept it, the framework stops fighting the architecture. The runtime can do what it is good at on regular regions and leave the hard ownership surfaces explicit. That is not a compromise. It is what real scalability looks like when the model is not uniform.

## References

- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- https://docs.pytorch.org/docs/stable/checkpoint.html
