---
title: "Manual Splits and What They Cost"
date: 2026-04-18
author: MegaCpp Engineering
tags: [pipeline-parallelism, model-partitioning, megatron, moe, mamba]
summary: >
  Manual model splits can rescue difficult topologies, but they also hard-code
  operational debt. MegaCpp shows exactly where explicit boundaries help and
  where they become the wrong abstraction.
description: >
  A grounded look at explicit pipeline boundaries, pipe-delimited patterns,
  weighted partitioning, and the maintenance cost of forcing stage shapes by
  hand in hybrid attention, MoE, and recurrent stacks.
---

# Manual Splits and What They Cost

Manual splits are sometimes the only way to make a hybrid stack trainable, especially when attention, MoE, recurrent blocks, and auxiliary embeddings do not partition cleanly. But they carry a tax: every explicit stage boundary becomes a maintenance contract for RoPE state, embeddings, loss heads, metadata, and schedule assumptions. The right use of manual splits is tactical, not ideological.

When people discuss pipeline parallelism, they often skip over the part that actually breaks systems: deciding where the split points go. A clean transformer with repeated identical blocks can be partitioned by count. A real hybrid stack cannot. Once the model contains different block families, optional side inputs, expert-heavy layers, and recurrent segments, a naive equal split becomes a proxy for "hope the runtime will sort it out later." It usually will not.

That is why MegaCpp keeps a real path for explicit boundaries. The important detail is not merely that the model can be partitioned. The important detail is that the runtime acknowledges two different partitioning modes: automatic partitioning and pipe-delimited explicit boundaries in the pattern string. That makes the split decision visible, auditable, and debuggable.

## Why explicit boundaries exist at all

The pipeline runtime in MegaCpp does not hide what a stage really owns. `create_pipeline_stage` builds a stage from concrete layer spans, attaches embeddings to the first stage, attaches the head to the last stage, and wires in RoPE buffers, optional n-gram or structure embeddings, and stage-local window sizes. That is already enough to explain why manual boundaries matter. A stage is not just "some layers." It is a bundle of responsibilities.

The runtime contract around pipe-delimited `nem_pattern` supports this directly. If explicit PP boundaries are provided, the runtime expects the delimiter count to match the requested number of stages. That sounds strict because it is strict. Manual splits are effectively part of the program.

| Split mode | Benefit | Cost |
| --- | --- | --- |
| Automatic equal partition | Minimal setup | Ignores heterogeneous layer weight |
| Weighted automatic partition | Better first approximation | Still heuristic |
| Manual pipe-delimited boundaries | Exact operator ownership | Permanent maintenance burden |

The reason teams still use manual boundaries is simple: hybrid stacks can be very asymmetric. An `E` block with expert routing, a heavy `M` block, and a dense `A` block do not cost the same thing.

## Patterns like `AEMEAEMEAEMR` are more than notation

The pattern notation matters because it encodes where asymmetry comes from. In this naming, `A` means attention, `M` means Mamba-style state-space layers, `E` means expert/MoE, and `R` means recurrent. That is already more operationally useful than saying "a mixed architecture." It tells you why the split problem is hard.

A pattern like `AEMEAEMEAEMR` is not just a decorative label. It predicts that the partitioning problem will not be uniform across depth. Some stages will want attention-side buffers and RoPE-heavy behavior. Some will hit expert routing and dispatch collectives. Some will end with recurrent state handling. If you split that model only by raw layer count, you are pretending these blocks impose identical runtime cost. They do not.

The repo language around ablock, mblock, eblock, rblock, and cblock is useful here because it gives teams a practical vocabulary for stage composition. Even when the exact training schedule evolves, the notation helps preserve one key truth: the model contains qualitatively different segments, not just repeated blocks.

## The hidden cost is stage-local plumbing

The biggest mistake in discussions of manual splits is treating them as if the cost were only balancing FLOPs. In MegaCpp, the more persistent cost was plumbing. Every explicit boundary decides where side data must exist and where it must not.

The first stage owns token embedding and several optional input-side embeddings. The last stage owns the head and also receives auxiliary pieces such as MTP-related references when enabled. RoPE buffers are provided broadly enough to survive resume and split changes. Relation-bias handling needs access on all stages that compute the corresponding additive bias.

That means a manual split changes more than layer count. It changes where the runtime must preserve non-obvious state.

```yaml
# schematic pattern with explicit PP boundaries
nem_pattern: "AEME|AEME|AEMR"
pipeline_parallel_size: 3
weighted_pipeline_split: true
```

The value of an explicit pattern like this is that it makes the decision inspectable. The cost is that every future modifier now has to preserve its assumptions.

| Stage concern | Why manual split affects it |
| --- | --- |
| Token embedding | Must stay on the first stage |
| LM head | Must stay on the last stage |
| RoPE buffers | Need consistency across resumes and stage changes |
| Structure or platform embeddings | First-stage ownership matters |
| Relation bias | Can require cross-stage awareness |
| Aux losses / MTP hooks | Typically last-stage anchored |

This is why manual splits often feel worse over time than they did on day one. The initial change is easy. The ongoing burden is keeping all of this aligned as the model evolves.

## Weighted automatic partitioning is better than equal split, but not enough

The runtime also includes weighted partitioning for MoE-aware layouts. That is important because it shows the team did not jump straight from naive splitting to hard-coded splits. There is a middle ground: heuristics that recognize some layers cost more than others.

That middle ground is useful, but it does not eliminate the need for explicit boundaries. Weighted partitioning helps when asymmetry is broad and predictable. It helps less when the topology itself matters. For example, if a stage needs to end before a recurrent transition, or if you want expert-heavy blocks clustered away from a fragile communication boundary, the problem is not just weight. It is semantics.

So the real decision tree looks like this:

1. Start with automatic partitioning when the model is homogeneous enough.
2. Use weighted partitioning when different block families have meaningfully different cost.
3. Use manual boundaries when topology or ownership rules matter more than heuristics.

That ordering is important because manual splits should be the last resort that remains explicit, not the first tool used out of habit.

## Manual boundaries can also freeze old assumptions

Another cost of explicit splits is that they preserve history, not just intent. A boundary that made sense before a model changed may become the wrong boundary after side features, compile strategy, or expert implementation change. The pipeline code comments about interleaving and schedule equivalence are a reminder that schedulers evolve. What counted as a balanced stage at one point can become a bad stage later.

The same is true for MoE behavior. The expert path in MegaCpp includes variable-split dispatch, compile-disabled outer routing, and logic that avoids padded equal-split behavior where it wastes memory. If the cost profile of eblocks changes, the old split might still be syntactically valid while being operationally bad.

This is the tax manual splits impose: they make topology explicit, but they also make topology sticky.

## What manual splits are good for

Despite all of that, manual splits are not a mistake. They are often exactly the right tool in hybrid systems. They are good for preserving intentional stage ownership when the model has non-uniform blocks. They are good for aligning expensive collectives away from fragile boundaries. And they are good for making the partitioning decision inspectable when debugging a pipeline schedule.

The key is to be honest about the price. Manual splits are a control surface, not a simplification. Every explicit delimiter in a pattern string is a promise that the surrounding runtime assumptions still hold.

That is why the best use of manual splits in MegaCpp was not "we prefer hand tuning." The best use was "the model is heterogeneous enough that implicit heuristics are no longer a sufficient explanation."

## The real cost is organizational, not just technical

The deepest cost is that manual splits create long-lived operational knowledge. New contributors have to understand why the split exists, what side effects it protects, and which invariants they must re-check when adding a new block family. If that knowledge is not written down, the split degrades into superstition.

That is why pattern notation and explicit stage construction matter so much. They convert a hidden arrangement into a debuggable contract. The contract is still costly, but at least it is legible.

In practice, that is the trade worth making. When the stack is simple, let heuristics win. When the stack becomes `AEMEAEMEAEMR` with real ablock, eblock, mblock, and rblock asymmetry, use manual splits deliberately and assume they are part of the architecture, not just part of the launch script.

## Schedule mechanics are part of the split cost

The comments in the pipeline runtime around virtual pipeline parallelism are a useful reminder that a split is also a schedule decision. The implementation references Megatron-style interleaving and the relationship between schedule tables, forward stage index, and rounds of microbatches. That means a manual split is never only about parameter placement. It also affects how bubbles, warmup, and flush behavior are experienced by the actual runtime.

This becomes especially important in heterogeneous models. If one stage contains an eblock-heavy segment and another stage contains lighter ablocks, a schedule that is technically valid can still create synchronization pressure or idle windows that are hard to diagnose just from high-level metrics. Manual splits can correct that, but the correction itself becomes one more piece of scheduling knowledge that has to stay alive across future changes.

## Resume, refactor, and feature growth all make old splits worse

A split that is correct for one revision can become subtly wrong after a refactor. The stage builder comments about resuming from a different split are especially revealing because they admit the problem directly. Some buffers and references are passed broadly not because that is elegant, but because the runtime needs enough continuity to survive stage-layout changes.

That is the long-term operational cost of manual boundaries. They pin architectural intent at a moment in time. If later work adds new side embeddings, changes recurrent-state handling, or moves an auxiliary loss surface, every explicit split has to be re-evaluated. Otherwise the code still runs, but the split becomes a fossil from an older model.

The right habit is to treat manual splits as versioned architecture, not a temporary launch tweak.


## References

- https://github.com/DatasunriseOU/site_samples/tree/main/docs
- https://github.com/DatasunriseOU/site_samples/blob/main/examples/hybrid/hybrid_pattern_sample.py
- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/pipeline_parallel.html
