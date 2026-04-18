---
title: "Sequence, Context, and Expert Splits in the Hybrid Stack"
description: "A concrete guide to what SP, CP, TP, and EP actually touch in the hybrid training stack, what communication each one introduces, and what each split is structurally forbidden from touching."
date: "2026-04-18"
tags: ["sequence-parallel", "context-parallel", "expert-parallel", "tensor-parallel", "moe", "mamba3", "training"]
---

TL;DR: TP, SP, CP, and EP are not four ways to say “make it fit.” They partition different objects. TP shards matrices and heads, SP shards activation flow on the TP axis, CP shards long-context ownership for attention-style paths, and EP shards expert ownership plus routed-token transport. Each one helps a different bottleneck, and each one is structurally unable to solve the others’ problems.

## Why MegaCpp cares

The specialist runtime is hybrid: attention blocks, Mamba blocks, and MoE blocks coexist in one model. That means distributed axes have to respect block semantics rather than just tensor shapes. The wrong mental model leads directly to wrong launch plans and wrong debugging priorities. If you think EP should shrink dense attention, or CP should automatically work for a recurrent Mamba path, you are already off the rails.

The prototype pins the intended story down in the distributed parallelism module. Its top-level docstring says the orchestrated order is `TP -> PP -> FSDP2 -> EP`, and its validation logic says TP must divide heads while EP must divide the number of routed experts. Those two constraints alone tell you that the axes are operating on fundamentally different model objects.

## What we built in the POC

Start with TP. In the distributed parallelism module, TP is applied first because it shreds layer-local matrices before later wrappers are added. The validation path checks that TP divides `num_heads` and `n_kv_head`, which is a direct statement about what TP is for: head-structured and matrix-structured modules such as QKV projections, output projections, MLP projections, and related shard-aware math. TP is not a generic model splitter. If a tensor is not written as a TP-aware matrix surface, TP will not save you.

That point matters especially in hybrid models because the presence of several block families can make TP look more universal than it is. It is not. TP can help at the projection edges of a Mamba block because those are still matrices. It cannot automatically partition the internal sequence-state logic the way it partitions a QKV projection. When people say “the model is TP=2,” what they really mean is that the TP-aware matrix surfaces are split by two and the rest of the runtime adapts around that fact.

SP is related to TP but not identical. the distributed parallelism module says the CUDA TP path enables sequence parallelism on the TP axis when `tp > 1`. That means SP is an activation-memory optimization that lives around TP-sharded compute. It shards the sequence dimension for the LayerNorm, dropout, residual, and related activation regions so those surfaces do not stay fully materialized on every TP rank. `CHANGELOG.md` captures the payoff: SP reduces activation memory proportional to TP degree. It also records one of the real failure modes: an SP reduce-scatter dtype mismatch with FP8 activations. That bug only makes sense if SP is understood as activation traffic on the TP axis rather than as a general-purpose parameter split.

CP is a context split, which is a different animal again. CP divides ownership of the token timeline so long-context attention can reconstruct full context with inter-rank exchange. It is useful when sequence length itself is the dominant pressure. But because CP cuts along time, it only makes sense for blocks whose semantics can be reconstructed from context-partitioned state. That is natural for ring-style attention and much less natural for recurrent state-space paths.

This is the split that most often gets overgeneralized. Engineers see token-shaped tensors on every block boundary and assume that any token-shaped representation can be context-partitioned. The prototype’s hybrid structure is a good corrective. Attention is fundamentally about interactions over context windows, so a dedicated context reconstruction path is natural. A recurrence is fundamentally about carrying temporal state, so cutting the timeline is a semantic operation, not just a layout change.

The hybrid stack makes that limit obvious. Mamba carries recurrence across sequence chunks. A CP split that simply chops the timeline without carrying the right boundary state would be wrong even if tensor shapes still line up. So CP is not “SP but on another axis,” and it is not a free answer for every token-shaped tensor in the model.

EP is the clearest special-purpose split. the distributed parallelism module says EP exists for MoE expert weights and that tokens are dispatched via all-to-all between EP peers. the MoE dispatch runtime module then shows the mechanism in detail: owner-rank mapping, send and receive counts, `all_to_all_single`, compact token receives, and combine-side returns. EP therefore partitions expert ownership and routes tokens to the ranks that own the selected experts. It does not reduce dense attention memory. It does not split sequence length. It does not fix a head-divisibility problem. It is specific to expert banks and routed-token transport.

That specificity is exactly why EP composes well with other axes instead of replacing them. EP can coexist with TP because one answers “who owns this expert bank?” while the other answers “how is this matrix factored across ranks?” EP can coexist with SP because one moves sparse token traffic to experts and the other shrinks dense activation flow around TP regions. Once you see each split as a different ownership contract, the composition becomes much less mysterious.

Those distinctions are much easier to keep straight if you ask one question for every split: what object is being partitioned?

| Split | Partitioned object | Main communication | Best use | What it cannot fix |
| --- | --- | --- | --- | --- |
| TP | shardable matrices and head groups | all-reduce or reduce-scatter in TP-aware modules | dense projection weight ownership | expert ownership, arbitrary timeline splits |
| SP | activation sequence slices on the TP axis | scatter, gather, reduce-scatter | activation-memory pressure around TP compute | parameter count, expert banks |
| CP | long-context token ownership | context exchange or ring-style attention traffic | long attention contexts | arbitrary recurrent state |
| EP | expert banks and routed-token destinations | all-to-all dispatch and combine | MoE parameter and compute ownership | dense non-MoE layers |

The concrete examples in the stack line up cleanly with that table.

An attention QKV projection is a TP target. SP can reduce the surrounding activation footprint. CP can matter if that attention block is handling long context. EP is irrelevant.

A routed expert bank in an EBlock is an EP target. TP may still apply inside the expert implementation if the expert math is TP-aware, but EP decides who owns the expert. SP still helps the surrounding activation path. CP does not change who owns the expert bank.

A Mamba recurrence is the opposite: TP can split matrix edges into and out of the block, SP can reduce some surrounding activation surfaces, but CP cannot casually cut through the recurrence and still claim correctness.

An embedding table is the reminder that none of these splits is magic. SP, CP, and EP do not make replicated parameter surfaces disappear. If the estimator says something is replicated, you still have to pay for it.

That negative space is worth emphasizing because it is what keeps launch planning honest. Every split description should have a companion statement about what it cannot touch. If it does not, people start imputing benefits from the wrong axis and then wonder why the measured memory or communication profile does not match the pitch.

the distributed parallelism module also makes the application order do real explanatory work. TP first because matrix ownership must be established before anything else. PP next because it partitions whole layers. FSDP2 next because it shards stage-local state. EP last because it needs the MoE routing substrate already in place. If you describe the system in any other order, you start crediting the wrong split for the wrong effect.

```python
config = ParallelismConfig(pp=..., dp=..., tp=..., ep=...)
validate_3d_config(config, model_config)
meshes = build_3d_mesh(config, device_type="cuda")
result = apply_3d_parallelism(model, config, meshes, device=device)

# ownership order in the runtime:
# TP -> PP -> FSDP2 -> EP
```

## How it lands in MegaCpp

The production lift is mostly about preserving this split taxonomy in user-facing configuration and internal debugging.

SP should remain described as TP-axis activation sharding, not as a general sequence-splitting feature. CP should remain described as a long-context attention strategy, not as a universal token partition. EP should remain described as expert ownership and routed-token transport, not as a generic shrink-ray for the whole model. And TP should remain tied to matrix and head divisibility constraints.

That matters operationally. A reduce-scatter issue around norms and FP8 activations is an SP/TP problem. A routed-token imbalance or an all-to-all hang is an EP problem. A long-context attention reconstruction bug is a CP problem. The right diagnosis starts by naming the split correctly.

The same is true for feature planning. If the next architecture change is to increase long-context ambition, CP and the attention backend surface become first-class concerns. If the change is to expand MoE capacity, EP ownership and routing pressure become first-class concerns. If the problem is plain dense activation pressure, SP is the first thing to inspect before inventing new complexity elsewhere.

## Ablations and what we kept

The changelog surfaces are consistent with that reading.

We kept SP because `CHANGELOG.md` says it reduces activation memory proportional to TP degree. We kept TP because the rest of the hybrid stack depends on shardable projection ownership. We kept EP because the MoE path needs real expert distribution rather than pretending every expert is local. And we kept the Mamba caution because recurrent sequence state is not the same as attention context.

We also kept the dtype and collective lessons. The recorded SP FP8 reduce-scatter mismatch and TP all-reduce placement fixes are exactly the sort of bugs that appear when the split taxonomy is real rather than rhetorical.

Those fixes are a useful reminder that the split map is not just about memory. It is also the shortest path to the right communication and correctness surface. A bug around head placement, reduction dtype, or routed-token traffic usually tells you immediately which axis deserves your attention if the taxonomy is clear.

## Public checklist

- Use TP only on layers whose matrix and head geometry actually supports it.
- Treat SP as TP-local activation sharding, not as a replacement for CP.
- Use CP only on blocks whose sequence semantics can be reconstructed correctly.
- Use EP only for expert-bank ownership and routed-token transport.
- Debug collective failures by split type instead of by vague “distributed” labels.
- Preserve the runtime order `TP -> PP -> FSDP2 -> EP` when reasoning about ownership.

## References

- the distributed parallelism module
- the expert-dispatch and dense-model runtime notes
- the recurrent-hybrid implementation notes
- the public Mamba fused trapezoidal kernel sample
- public changelogs for split-policy changes
