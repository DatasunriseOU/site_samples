---
title: "Training speed by feature: which parts of the stack really move step time"
description: "A grounded feature-by-feature look at training-speed impact across a modern hybrid stack: Mamba fused paths, memory-traffic cleanup, MLA pieces, MoE dispatch, routing bridges, and feature taxes that should stay experimental."
date: "2026-04-18"
tags: ["performance", "kernels", "mamba3", "moe", "mla", "transformer-engine"]
---

# Training speed by feature: which parts of the stack really move step time

Not every interesting feature moves training speed in the same way. The biggest durable wins usually come from removing repeated hot-path work: fused Mamba or state-space updates, fused residual math, narrow MLA ingress fusion, and especially MoE dispatch-plus-compute cleanup. Some features are speed enablers because they route work to a better backend. Others are quality or architecture features that should be treated as measured taxes, not presumed accelerators. The practical job is to separate hot-path wins from feature costs and to keep exact measurements for both.

The easy way to talk about speed is to say everything matters. The useful way is to ask a narrower question: which code paths are executed often enough, over tensors large enough, that cleaning them up changes the wall-clock reality of training? A serious training stack answers that question in a grounded way. It exposes throughput and goodput reporting, preserves feature flags in launch surfaces, and keeps a reproducible record of which optimizations are worth keeping versus which ones are still experiments.

## Start from observability, not from intuition

A stack can only have a sane speed conversation if it measures the right thing. The local repos already have the right ingredients: goodput accounting, temporal performance reporting, and machine-checkable result records. That matters because otherwise every feature discussion degrades into profiler screenshots and memory of “it felt faster.”

| Surface | What it measures | Why it matters for feature evaluation |
| --- | --- | --- |
| Goodput accounting | Useful training time versus badput categories | Separates model progress from compile, idle, or data overhead |
| Temporal performance tracking | Step-level throughput, tokens, and peak memory | Shows feature tax or gain over time instead of at one lucky step |
| Stable report output | Comparison-ready summaries | Makes comparison shareable instead of anecdotal |
| Structured result schema | Structural invariants for results | Prevents incomplete ablations from being treated as final |

That measurement layer changes the whole conversation. Once it exists, features stop being judged by enthusiasm and start being judged by whether they improve useful work per unit wall time, and whether they do so reliably.

## The hot-path wins are the ones worth defaulting on

The strongest speed features in the current tree share one property: they eliminate repeated work inside paths that dominate the training loop.

Mamba-related fused work fits that pattern. In hybrid lanes where `M` blocks are active, any fused state update or scan cleanup hits a core loop, not a side branch. That is why Mamba-side fusion belongs in the “likely worth keeping” bucket. If the model uses many Mamba layers, repeated elementwise and state-update overhead compounds quickly.

The same logic explains the value of residual-path fusion. Small-looking operations matter because they run constantly. Elementwise launches on activation-shaped tensors are easy to dismiss one by one and expensive in aggregate. When those are fused into fewer passes over memory, the gain is rarely dramatic at one instruction boundary and often very real end to end.

MLA is a more mixed case, but still instructive. Narrow ingress fusion is the kind of optimization that often survives scrutiny because it removes reshape/apply/reshape style overhead from a hot boundary. Broader projection fusion, by contrast, is something to treat more skeptically until it continues to beat library improvements and compiler evolution. The general lesson is not “MLA fusion is good” or “MLA fusion is bad.” It is “keep the narrow wins that repeatedly hit hot ingress paths, and force the bigger fusions to prove themselves.”

## MoE is the largest obvious speed surface

If one feature family deserves to be called a first-order throughput concern, it is MoE. A fused MoE implementation makes that visible by comparing a standard route-permute-pad-batched-gemm-unpad-unpermute shape against a tighter route-sort-jagged-gemm-weighted-scatter shape.

```text
standard:
route -> permute -> pad -> batched_gemm -> unpad -> unpermute

fused:
route -> sort_by_expert -> jagged_gemm_fused -> weighted_scatter
```

That is not just an implementation detail. It is a map of where speed disappears. Every extra permute, pad, and unpad stage is another opportunity for memory traffic and dispatch overhead to dominate the useful expert compute. When MoE underperforms, the culprit is often not the GEMM itself but the work around it.

This is also where EP and speed meet directly. Expert parallelism is not only a scaling feature. It changes the cost model of the model. Routing, sorting, combine, and cross-rank ownership are part of step time. That is why the combined `TP + SP + EP + FSDP2 + compile` lane matters for performance too. If the lane only barely works semantically, the throughput number will be meaningless. Once the lane is healthy, MoE optimization becomes one of the highest-payoff speed investments in the stack.

Public-facing architecture notes reinforce this. MoE, grouped GEMM, DSA, and MTP should stay explicitly visible in topology and throughput discussions. That is the correct shape: MoE is not a feature you mention in model prose and then ignore in speed accounting.

## Some features are backend selectors, not direct kernels

Another important category is the backend bridge. These are features that may not themselves be a new fused kernel but still matter because they route work onto a better-maintained fast path. In production engineering, that can be as valuable as writing a new kernel by hand.

Dispatcher boundaries serve that role. The production lesson is straightforward: centralize backend choice when possible. If a better vendor path exists for a narrow operation, the right architecture is often a disciplined dispatcher with solid fallback behavior, not hardwired backend-specific branches scattered across model code.

This matters for speed because it changes the maintenance cost of staying fast. A dispatcher can inherit improvements from upstream backends. A custom path has to keep justifying itself against that moving baseline.

## Feature taxes should be treated honestly

Not every feature is supposed to be a speed win. Some are architecture or quality features that impose extra compute, memory, or bookkeeping. The mistake is not that they exist. The mistake is pretending they are “free enough” without measurement.

In this bucket belong things like STP and other auxiliary-loss or metadata-heavy features. They may be good ideas. They are not baseline speed features. The correct question is not “are they elegant?” It is “what do they do to throughput, memory, and convergence-adjusted productivity?”

| Feature family | Likely speed role | Default stance |
| --- | --- | --- |
| Fused Mamba path | Direct hot-path speed win | Default on when the model uses it |
| Fused residual helpers | Repeated small wins that compound | Default on |
| Narrow MLA ingress fusion | Direct local win | Default on if validated |
| Broad MLA projection fusion | Conditional / needs repeated proof | Keep selective |
| Fused MoE dispatch and compute | Major throughput lever | Treat as core optimization |
| Backend bridge / dispatcher | Indirect speed enabler | Keep centralized |
| STP and similar aux features | Measured tax | Keep experimental |

That table is the practical decision surface for a production stack. The core rule is easy: hot-path wins can graduate to defaults; taxes must keep proving their value.

## NAM56R is a good example of why feature accounting matters

The NAM56R family is a good illustration because it concentrates several feature families in one model description: hybrid block patterns, MoE, DSA, MTP, and hardware-specific throughput claims. Public recipe samples preserve the pattern layer, while public status notes keep measured configurations and throughput on H200-class systems visible.

That means “training speed by feature” cannot be separated from “training speed by model shape.” A feature that is minor on a dense lane can become a first-order concern on a hybrid `AEMEAEMEAEMR` lane. A DSA optimization that matters when full and shared layers interleave may be irrelevant on a simpler topology. The only sane answer is to keep exact model naming, exact topology, and exact feature state in the same measurement record.

An index-cache optimization is a good example of a feature whose value depends on shape. The patch exists because adjacent DSA layers share enough top-k structure that recomputing indexer work every time is wasteful. That is not a universal speed truth. It is a shape- and architecture-aware optimization. But when the pattern fits, it is exactly the kind of repeated overhead reduction worth keeping.

The same goes for a streamlined MTP layer. It is not just “MTP exists.” It is a narrow design that bypasses a more complex path and avoids some SP/TP workspace burden in the shared block. That is a speed-relevant implementation choice, not just a feature flag.

## How to decide what graduates into production

The production rule should be conservative and repeatable.

1. Keep features that remove repeated work from the inner training loop.
2. Prefer narrow, validated fusion over giant fused abstractions that may age badly.
3. Centralize backend selection rather than scattering backend-specific logic.
4. Treat quality or architecture features as opt-in taxes until repeated measurements say otherwise.
5. Record every meaningful ablation with exact model names, topology, and observability output.

That rule sounds procedural because it is. Most bad speed decisions happen when a team skips procedure and promotes a feature because it sounds important.

```yaml
speed_defaults:
  fused_mamba: true
  fused_residual: true
  fused_mla_ingress: true
  fused_moe: true
  backend_dispatch: auto
experimental_taxes:
  stp: false
  extra_aux_losses: false
observability:
  goodput: true
  temporal_perf: true
  measurements: true
```

That config is illustrative. It captures the right posture: enable the repeated hot-path wins, keep dispatcher logic on, and force experimental taxes to justify themselves.

## The durable lesson

The durable lesson is that speed is rarely improved by a single giant trick. It is improved by repeatedly removing unnecessary work from the surfaces the model hits every step, then preserving enough measurement context that the gain can survive handoff and re-testing.

That is why the local measurement layer matters as much as the kernels. Without it, the team cannot tell whether a feature is a real acceleration, a small tax with quality upside, or just a one-run illusion.

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/docs/hybrid-layout-notes.md
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md
- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html
