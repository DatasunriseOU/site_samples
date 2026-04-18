---
title: "FSDP2 Pain and Payoff: What Actually Reduced Memory"
date: 2026-04-18
author: MegaCpp Engineering
tags: [fsdp2, pytorch, distributed-training, memory, mixed-precision]
summary: >
  What the research repo and MegaCpp code show about FSDP2 in practice: where
  selective sharding helps, where it backfires, and why the stable policy is
  narrower than the easy marketing story.
description: >
  A code-grounded look at FSDP2 rollout choices: selective wrapping,
  reshard policy, mixed precision, expert edge cases, and the interaction with
  TP, SP, EP, compile, and checkpointing.
---

# FSDP2 Pain and Payoff: What Actually Reduced Memory

**TL;DR:** FSDP2 helped the POC when it was treated as a narrow ownership tool, not as a blanket switch. The grounded win was selective wrapping at real stage or block boundaries, explicit mixed-precision policy, and conservative reshard behavior once deep stacks, expert routing, and compile were all active. The grounded loss was over-wrapping, then compensating with more exceptions, more optimizer special cases, and more fragile interaction between sharded parameters and runtime-specific paths.

The easy story about FSDP2 is still attractive: shard parameters, gather them for compute, then scatter gradients on the way back. That summary is directionally right, but it hides the operational question that actually determined whether memory improved: what exactly owns the live parameter state at each boundary of the forward and backward passes?

The research repo makes that question unavoidable because its runtime is not a single uniform transformer stack. It mixes attention, expert layers, recurrent or Mamba-style layers, specialized dispatch paths, compile lanes, and pipeline-stage composition. The MegaCpp production side tells the same story from the launcher and recipe angle: NAM56R is expressed as a patterned model, `AEMEAEMEAEMR`, with materially different block families mapped to different runtime behavior. Once a stack is heterogeneous like that, FSDP2 stops being a binary choice and becomes a placement problem.

That is why the strongest lessons are narrower than the popular blog-post version. FSDP2 paid off where the code had large, repeatable weight-bearing surfaces and predictable collective ownership. It hurt where the wrapped region crossed too many runtime seams, especially optimizer, compile, and expert-dispatch seams.

## Why selective wrapping beat global wrapping

One of the clearest clues is in the POC test and rollout notes: `apply_cuda_fsdp_to_pp_stages()` was updated to wrap each `_PipelineStageModule` individually, and the notes call out that CPU unit tests passed for local stage coverage. That is a much more precise design than “turn on FSDP2 for the whole model.” It says the stable unit of sharding is not the Python process and not the entire module tree, but the stage-sized ownership boundary that already exists for pipeline execution.

This matters because stage boundaries are real runtime boundaries. They already decide activation flow, scheduling, and communication sequencing. When FSDP2 respects that shape, memory behavior becomes easier to reason about. When it ignores that shape, one optimization layer starts fighting another.

The model recipe side reinforces the same point. In the public NAM56R recipe sample, NAM56R is not described as a monolithic dense stack. It has explicit dimensions, explicit MoE defaults, explicit MLA defaults, and a pattern builder that maps `A`, `E`, `M`, and `R` symbols into Megatron-native layer symbols. That is a recipe-level acknowledgement that different layer families have different contracts. A sharding strategy that treats them as identical is already starting from the wrong abstraction.

| Surface | Stable FSDP2 posture | Why |
| --- | --- | --- |
| Pipeline stage wrapper | Strong candidate | Matches execution ownership and activation boundaries |
| Large dense projection blocks | Usually worth sharding | High parameter volume, predictable gather/scatter pattern |
| Expert-heavy regions | Worth sharding carefully | Big memory pressure, but only if dispatch ownership stays clear |
| Tiny helper modules | Often leave replicated | Low memory upside, high complexity tax |
| Mixed runtime seams | Avoid broad wrapping | Sharding contract becomes harder to prove |

This is also why the memory benefit looked real in some receipts and disappointing in others. The difference was rarely “FSDP2 on” versus “FSDP2 off.” It was whether the wrapped surface aligned with a real unit of compute ownership.

## The hidden issue was not sharding itself, but reshard timing

The next practical lesson is that reshard timing decided peak memory more often than the simple enable flag did. Even without reading every implementation detail, the repo state points to this clearly. The April 4 test-results note records live memory problems around FSDP2 AllReduceState FP32 buffers and `record_stream` over-allocation, and the recommended parallel lanes are framed in terms of PP plus FSDP2 combinations rather than a universal always-on recipe.

That is a signal that the dangerous case was overlapping live materialization. If several wrapped regions hold onto full parameter state longer than expected, then the theoretical benefit of sharding starts getting eaten by multiple simultaneous full views, reduction buffers, and overlap-related allocations.

In practice, the conservative answer was to prefer immediate cleanup of live full-weight views unless a very local exception was justified. That is why the mental model should be “make full parameters exist for the shortest trustworthy window,” not “avoid future all-gathers at any cost.”

```python
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

for stage in pipeline_stages:
    fully_shard(
        stage,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
```

That block is schematic, but it captures the stable policy shape reflected in the repo: explicit precision, explicit stage-level wrapping, conservative resharding.

| Policy choice | Upside | Failure mode |
| --- | --- | --- |
| `reshard_after_forward=True` | Better peak memory discipline | More future gathers |
| `reshard_after_forward=False` | Potentially fewer gathers | Full views pile up across deep stacks |
| Local exceptions only | Lets critical roots stay hot | Requires real evidence, not guesswork |

Once TP, SP, EP, and compile enter the picture, the cost of a bad no-reshard assumption compounds quickly. The repo’s own rollout notes make that visible by recommending specific PP+FSDP2 combinations rather than pushing a single universal setting.

## Optimizer behavior changed the real economics

Another under-discussed part of the FSDP2 story sits in the prototype model runtime file. The optimizer setup is not naive. It contains explicit detection for shard-backed parameters, routes some parameter classes differently, and introduces `FSDP2Muon` for the case where local shard-backed matrix parameters need optimizer behavior that respects the sharding contract.

That is important because memory is not just a question of model weights. The optimizer can silently claw back much of the benefit through fp32 master copies, extra state, or wrapper-specific staging. The comments around these paths make the repo’s position clear: plain optimizer assumptions were not trusted once parameters could be shard-backed DTensors or otherwise wrapped by a distributed runtime.

This is one reason the rollout narrowed over time. A broad FSDP2 rollout often forces a second broad cleanup in optimizer ownership, because the old optimizer assumptions were written for replicated tensors. If that second cleanup is not done, you get a partial win on model state and a partial loss somewhere else.

The POC therefore took the harder but more honest route: distinguish shard-backed matrix params from plain params, then route them accordingly. That design is less elegant in the abstract and more correct in practice.

## Why expert parallelism complicated the picture

The NAM56R and MoE-related code paths explain why expert regions need extra care. In the recipe layer, the model is not only hybrid but explicitly MoE-aware: `MOE_NUM_EXPERTS = 16`, `MOE_TOPK = 4`, `MOE_FFN_HIDDEN = 896`, and `MOE_SHARED_EXPERT_SIZE = 1024` are part of the declared grounded spec. In the runtime, the dedicated expert-routing and dispatch modules exist because routing, permutation, dispatch, and combine are not incidental details. They are first-order runtime structure.

FSDP2 interacts with that structure in two ways.

First, the expert bank itself is a real memory target. Sharding it can be worthwhile because expert parameters are large and numerous. Second, the dispatch path adds metadata, active-token selection, combine buffers, and backend-specific launch behavior. If those parts are not cleanly separated from parameter ownership, the memory savings from sharding can be diluted by new transient buffers.

That is exactly why the kernel-substrate decision document is useful in a post about FSDP2. It states that the implementation target is not merely to reduce autograd saved state; it is to reduce runtime materialization and HBM traffic end to end, including routing buffers and activation materialization. The moment you read that, you can see why “just add FSDP2” was never going to be the full answer for expert-heavy lanes.

| Expert-parallel question | FSDP2-friendly answer | Anti-pattern |
| --- | --- | --- |
| Where do we shard? | Large expert parameter surfaces | Every helper around the dispatch stack |
| What stays explicit? | Routing, dispatch, combine ownership | Hidden side buffers and implicit conversions |
| What is the target? | Lower end-to-end materialization | Only lower saved tensors in autograd |

The stable lesson is that FSDP2 is not a substitute for dispatch cleanup. In expert-heavy models, it is one lever inside a larger memory-budget plan.

## Compile and pipeline overlap made bad assumptions visible

The April 4 feature test note is especially revealing because it frames recommended configurations as `1F1B PP=2 + FSDP2 dp=4`, `DualPipe PP=4 + FSDP2 dp=2`, and `DualPipeV PP=4 + FSDP2 dp=2`. That is not merely a launcher matrix. It is evidence that once overlap and pipeline scheduling matter, the “best” FSDP2 posture depends on how much of the model each rank owns and how much overlapping activity happens at once.

Compile made this more obvious, not less. The repo has historical H200 FSDP2 receipts tied to a compatibility path that appends `--fsdp_cuda --regional_compile --activation_memory_budget=1.0`. Even without retelling the entire bring-up history, that shows that compile was not neutral. It changed the acceptable memory regime enough that FSDP2 cases needed a compatibility lane.

That should update how people talk about FSDP2 in modern training systems. The right question is not “does FSDP2 reduce memory?” It does. The right question is “under this schedule, optimizer, compile mode, and module partitioning, what remains live at the same time?” That is the question the repo kept answering more narrowly over time.

## What the payoff really was

The payoff was not magic and it was not fake. It was narrower and therefore more useful. FSDP2 gave the POC a way to put large parameter surfaces on a disciplined ownership regime, especially when combined with stage-aware wrapping and explicit precision. It also created a path for viable PP+DP compositions that would otherwise struggle with replicated parameter pressure.

But the pain was equally real. Broad wrapping introduced optimizer routing work. Expert paths exposed transient-buffer reality. Compile changed the safe operating envelope. Pipeline overlap made retained materialization more expensive. And the memory story only stabilized once the team stopped asking for a single switch and started reasoning about block families and stage boundaries.

That is why the cleanest summary is not “FSDP2 works” or “FSDP2 is overhyped.” The grounded summary is this: FSDP2 works when it is allowed to be specific.

## References

- a feature-validation results note
- an H200 modal matrix workflow
- the main model runtime module
- the main MoE runtime module
- the MoE dispatch runtime module
- the public NAM56R recipe sample
- the public NAM56R launch sample
