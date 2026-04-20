---
title: "H200 Memory Geometry for the Hybrid Stack"
description: "How weights, gradients, optimizer state, activations, routing scratch, runtime reserve, and fragmentation stack up on one H200 device in a hybrid training stack."
date: "2026-04-18"
tags: ["H200", "memory", "muon", "moe", "mamba3", "activations", "training"]
---

On H200, memory fit in a hybrid stack is decided by geometry, not by a single parameter-count number. A useful capacity model splits the budget into replicated weights, tensor-parallel shards, expert-parallel shards, gradients, optimizer state, activations, routing scratch, runtime reserve, and allocator overhead, and that split is the only reliable way to predict whether a launch actually clears the HBM limit.

## Why MegaCpp cares

The specialist stack is a hybrid of attention blocks, Mamba blocks, and MoE blocks, so one device budget is shaped by several different ownership rules at once. Some tensors replicate everywhere, some split across TP, some split only across EP, and some exist only during backward or serving. That means “the model is about this big” is not enough for launch planning. The usable question is how the full memory geometry lands on one GPU after TP, EP, DP, optimizer assignment, and recompute policy have all taken effect.

The pre-flight answer should live in a capacity model that names what it counts: dense layers, recurrent or state-space blocks, MoE blocks, activation checkpointing, sharding strategy, and precision format. On H200, a realistic assumption is about 141 GB of usable HBM per device after reserve and allocator behavior are accounted for. Precision choices such as BF16, FP8, and low-bit weight formats then set the byte cost of each tensor family. That is a much better starting point than a spreadsheet approximation.

## What a useful estimator should expose

The central output should track categories such as replicated parameters, tensor-parallel parameter shards, expert-parallel parameter shards, gradients, optimizer state, activations, routing scratch, feature-specific activations, allocator overhead, and runtime reserve. That is more useful than a single total because it exposes which axis is actually dominant.

The first split is parameter ownership. Embeddings and norms are usually replicated. Attention, MLP, and recurrent projection weights are often sliced by tensor parallelism. Expert banks are distributed by expert parallelism. A common ordering is tensor parallel first, then pipeline partitioning, then state sharding, then expert placement. That ordering is the reason a serious estimator does not pretend every parameter shrinks with every degree.

That point is easy to miss in practice because hybrid models encourage people to talk in aggregate counts. But the runtime does not load an aggregate count. It loads concrete tensor families with different ownership rules. Embeddings, routers, and norms remain visible on every relevant rank. Attention projections can be sliced by TP because their algebra was written for it. Expert banks can be handed to EP because token dispatch later reconstructs the ownership boundary at runtime. The geometry is therefore not just a memory story. It is the consequence of what the execution model is legally allowed to split.

The second split is optimizer geometry. A stack does not pay one flat optimizer tax. Large matrix paths, embeddings, norms, routers, and other special-case tensors often carry different optimizer state. That matters because one change in parameter grouping can swing the optimizer column even when the raw parameter count does not move much.

The third split is the activation surface. In the estimator, ordinary activations, MoE routing state, and feature activations are distinct fields. In the runtime, that matches reality. The dense-model path carries residual and normalization state through attention blocks; the MoE dispatch path builds dispatch counts, rank-local receives, and combine buffers for the MoE route; and the feature side adds smaller but still real activation surfaces for optional structure-aware paths. That is why this category has to be managed explicitly: sequence parallelism reduces activation memory roughly with tensor-parallel degree, and selective activation recomputation is often a better lever than blunt full checkpointing when the goal is to shrink the activation column without overpaying in extra compute.

This is also where hybrid depth matters more than people expect. A stack with more attention-heavy layers pays for a different activation mix than a stack with more Mamba or expert-heavy layers, even when total parameter count is similar. Dense attention-side residuals dominate one profile, routed-expert metadata and dispatch scratch dominate another, and sequence-state work dominates a third. The estimator's category split is valuable precisely because it lets us change architecture without losing the ability to reason about where memory is going.

The fourth split is routing scratch. Expert parallelism is not just expert parameters. The dispatch path also needs token counts, receive buffers, combine buffers, and sometimes extra transport metadata. Even with sparse expert math, the routing subsystem still allocates memory, which is why it deserves its own line rather than being smeared into activations.

The fifth split is the runtime tail. Any realistic estimate includes allocator overhead and runtime reserve, and it should warn when headroom gets too small. That is the estimator admitting that launch-time behavior is not purely steady-state math. Compile peaks, collectives, allocator scars, and scratch buffers create a real tail, which is exactly why the runtime tail stays visible in the model.

The allocator and collective tail is also where "safe on average" becomes unsafe in production. A launch can look fine if you only inspect final resident state after the step settles. But startup has to pass process-group construction, scratch allocation, compile-time graph work, and the first real backward. MegaCpp treats those as part of memory geometry rather than as unrelated incidents. That is a more honest model of how H200 jobs fail.

Serving adds one more dimension that training often does not pay: KV cache. The engine and serving stack distinguish contiguous cache and paged cache substrates, and only attention layers consume them. In a hybrid model that interleaves non-attention layers, that matters a lot. Mamba blocks do not contribute to KV residency the way attention blocks do, so the serving geometry of a hybrid stack is better than a pure-attention stack of equal depth.

The cleanest summary is to ask what scales each surface.

| Memory surface | Main scaling factors | Primary code surface | What it does not care about |
| --- | --- | --- | --- |
| Replicated params | width, vocab, replicated features | the capacity model | EP, SP, CP |
| TP-sharded params | width and TP degree | the parallelism plan | expert ownership |
| EP-sharded params | expert count, expert width, EP degree | the expert runtime and dispatch path | dense attention projections |
| Gradients | parameter geometry plus DP mode | the capacity model | serving cache layout |
| Optimizer state | optimizer assignment by tensor type | the capacity model | SP, CP |
| Activations | microbatch, sequence length, recompute policy | the dense-model runtime | EP on non-expert residuals |
| Routing scratch | routed tokens and top-k load | the expert-dispatch path | TP on dense-only layers |
| KV cache | serving batch and context length | the serving runtime | training-only backward state |
| Fragmentation and reserve | allocator and runtime behavior | estimator plus runtime-debug notes | any one sharding axis directly |

What H200 buys you is enough headroom that these categories can trade against each other instead of forcing a single desperate choice. Public change notes show the pattern directly: sequence parallelism shrank the activation term, selective recompute replaced full checkpointing, and a fused recurrent-convolution path removed a large transient-memory spike. Those were not isolated optimizations. They were geometry repairs.

Another way to say this is that H200 changes the optimization order. On a smaller device, you often begin with emergency compression or aggressive offload just to launch. On H200, you can instead choose the cleaner structural fix: better sharding of the right parameter family, a narrower recompute boundary, or a reduction in transient expert-routing scratch. That tends to produce systems that are easier to reason about and easier to carry into production.

```python
# The per-device budget surfaces that matter before launch.
Estimate(
    replicated_params_gb=...,
    tensor_parallel_params_gb=...,
    expert_parallel_params_gb=...,
    gradients_gb=...,
    matrix_optimizer_gb=...,
    other_optimizer_gb=...,
    activations_gb=...,
    routing_gb=...,
    runtime_reserve_gb=...,
    overhead_gb=1.5,
)
```

## How it lands in production

The production lift is to preserve the same memory geometry in launch planning and recipe review.

That means keeping replicated tensors explicit, keeping Muon and AdamW state separate, budgeting MoE routing scratch as its own term, and refusing to spend the full nominal HBM number as if runtime reserve did not exist. It also means respecting axis-specific ownership rules. Expert banks can be EP-sharded, but routers and embeddings do not disappear just because EP is enabled. Attention and Mamba projections can be TP-sharded, but the routing subsystem is a separate cost center.

It also means teaching the launch surface to answer the right questions. If a proposed recipe change increases expert count, the first follow-up should not be “what is the new parameter count?” It should be “which bucket grows: EP-sharded expert banks, routed-token scratch, or both?” If a precision change is proposed, the question is not just what happens to weights; it is what happens to gradients, optimizer state, communication buffers, and whether the runtime tail now dominates instead.

## Design choices that held up

Public code and documentation samples give the short list of memory decisions that held up.

The fused recurrent-convolution path stayed because it cut peak memory meaningfully compared with a naive grouped implementation. Sequence parallelism stayed because it reduces activation memory roughly with tensor-parallel degree. Selective activation recomputation stayed because it beat a blunt full-checkpoint strategy. Explicit routing accounting stayed because expert parallelism is not just “expert weights somewhere else”; the dispatch and combine path is part of the budget.

What we did not keep is the habit of talking about fit as if only parameters mattered. On H200, the winner is the configuration where weights, gradients, optimizer state, activations, routing scratch, KV cache, and reserve all fit together with real headroom.

That is the real meaning of memory geometry. It is not an aesthetic way to present a budget. It is a way to stop using a scalar mental model for a multidimensional runtime.

## Production checklist

- Run the estimator with the real TP, EP, DP, precision, and feature flags.
- Keep embeddings, norms, routers, and other replicated tensors visible in the budget.
- Track Muon and AdamW state separately when reviewing optimizer-group changes.
- Budget MoE dispatch and combine scratch whenever EP is active.
- Cross-check the activation estimate against the actual recompute policy in the dense-model runtime.
- Add KV cache for serving scenarios instead of assuming the training budget covers them.
- Leave nontrivial headroom for compile peaks and fragmentation on H200.

## References

- [NVIDIA H200 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h200/)
- [PyTorch `torch.utils.checkpoint`](https://pytorch.org/docs/stable/checkpoint.html)
- [Megatron Core developer guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [Distributed debugging notes](https://github.com/DatasunriseOU/site_samples/blob/main/docs/distributed-debugging-notes.md)
- [H200 training status summary sample](https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/docs/cppmega/training/training-on-h200-eight-gpu__production_status_summary__v1.md)
