---
title: "Expert Parallel and MoE Sharding: Capacity Is Cheap, Routing Is Not"
date: 2026-04-18
author: MegaCpp Engineering
tags: [expert-parallel, moe, distributed-training, sharding]
summary: >
  Expert parallelism scales routed-expert capacity, but real production
  constraints still live in router semantics, dispatch topology, auxiliary
  losses, and the exact layer contracts used by hybrid A/M/E/R models.
description: >
  A grounded walkthrough of expert parallelism in the MegaCpp stack, based on
  the recipe files, layer definitions, schedule plans, and bug reports that
  shape how MoE runs actually behave.
---

# Expert Parallel and MoE Sharding: Capacity Is Cheap, Routing Is Not

**TL;DR:** expert parallelism is the right primitive for scaling MoE capacity, but it is only one part of a working training lane. The hard engineering work lives in router behavior, dispatch backend selection, per-layer contracts, auxiliary and z-loss accounting, and the difference between CUDA process-group EP and XLA sharding. The code in the POC and the companion MegaCpp recipes is useful precisely because it does not flatten those distinctions.

When people first hear “expert parallel,” the mental model is usually simple: place different experts on different ranks, dispatch tokens to the right owner, and enjoy large parameter counts with modest active compute. That story is directionally correct, but it is incomplete in the same way that “tensor parallel splits matrices” is incomplete. The production cost is hidden in the contracts around the split.

The code makes those contracts explicit. In the main model runtime module, `EBlock` is not a vague concept; it is the point-wise expert family with its own MoE routing knobs, aux-loss terms, z-loss terms, shared-expert configuration, routing groups, FP8 options, and fused dispatch switches. In the public NAM56R recipe sample, the NAM56R recipe fixes the architecture schedule as `AEMEAEMEAEMR`, then maps only `E` symbols to MoE-bearing layers. In other words, EP is not a model-wide blur. It is a narrow contract attached to the `E` surfaces of a hybrid model.

## Where expert ownership actually begins

The cleanest place to start is the configuration surface. The POC’s `GPTConfig` separates layer families by responsibility: `ABlock` handles attention only, `MBlock` handles Mamba sequence mixing, `EBlock` handles expert FFNs, and `RBlock` handles recurrent or M2RNN-style sequence mixing. That separation matters because expert parallelism is meaningful only for one of those families.

The NAM56R recipe in MegaCpp preserves the same idea. `build_nemo_hybrid_pattern()` converts architecture symbols into Megatron-facing pattern markers, with `A -> *`, `M -> M`, `R -> M`, and `E -> E` when MoE is enabled. The recipe therefore tells the runtime, in a machine-checkable way, where expert ownership boundaries exist. That is a stronger contract than a prose architecture note, because the scheduler and launcher can use it directly.

| Concern | Owned by EP directly? | Evidence in code/docs |
| --- | --- | --- |
| Expert parameter residency | Yes | `moe_n_routed_experts`, `expert_tp_degree`, `expert_parallel` surfaces in config/runtime |
| Token-to-expert routing | Partly | `moe_routing_mode`, `moe_score_function`, group routing, dispatch backend choice |
| Shared expert execution | Partly | `moe_n_shared_experts`, `moe_shared_expert_overlap`, shared gate options |
| Aux and z-loss correctness | No | `moe_aux_loss_weight`, `moe_router_z_loss_weight`, pipeline-stage accounting still required |
| Cross-family layer scheduling | No | `ABlock`/`MBlock`/`EBlock`/`RBlock` are independent layer contracts |

This is the first corrective to over-simplified EP discussions: partitioning weights is the easy part. Keeping the routing, loss terms, and schedule semantics coherent is where runs become real or fragile.

## Why routing semantics dominate the real cost

`EBlock` in the POC exposes a surprisingly rich router surface. Routed experts can be scored with `softmax` or decoupled `sigmoid`. The routing mode can be `token_choice`, with older soft and expert-choice paths retained but clearly treated as legacy or discouraged. The config also includes group routing, loss-free load balancing, router z-loss, optional FP32 router math, input jitter, shared expert gates, and a fused-MoE path. None of those fields exists by accident.

They exist because the runtime cost of MoE is driven by more than how many experts there are. It is driven by what the router asks the system to do. Two configurations can both say “16 experts, top-k 4” and still stress the cluster very differently if one uses normalized token-choice routing with grouped selection while the other relies on a softer path that expands more compute or creates noisier exchange patterns.

The POC’s tests reinforce that point. sanitized MoE tests checks gradient flow through the router, checks linear experts, and checks selected-token execution. Those tests are not glamour pieces; they are acknowledgments that routing is part of correctness, not just a throughput concern. If the router contract is wrong, you do not merely lose a few percent of speed. You change which experts learn and whether the active path is even differentiating as intended.

That same idea appears in the historical reports. The live-bugs review notes that MoE-specific losses and resume semantics deserve explicit treatment, and the warmup regression report treats `regional_compile + MoE` as a separate risk class instead of pretending every compile lane is equivalent. Again, the codebase is telling you that an EP run is not just a dense run with a different sharding flag.

## EP in a hybrid A/M/E/R model is narrower than people expect

NAM52 and NAM56R notation helps because it keeps architecture reasoning honest. `A` means attention, `M` means Mamba, `E` means expert, and `R` means recurrent. In code-facing language those become `ABlock`, `MBlock`, `EBlock`, and `RBlock`. Related notes and code comments also use `ablock`, `mblock`, `eblock`, `rblock`, and sometimes `cblock` as shorthand. The point is that the model is heterogeneous by design.

That means EP does not relieve every memory bottleneck in a NAM56R lane. In `AEMEAEMEAEMR`, only the `E` slots are direct expert-ownership opportunities. Attention projections, latent caches, recurrent state, and Mamba-specific buffers remain separate concerns. If a run is tight on MLA cache pressure or recurrent-state residency, EP may still be valuable, but it will not solve that class of pressure.

The MegaCpp scheduler code makes this sharper. the public hybrid schedule sample explicitly distinguishes MoE transformer layers from opaque non-MoE layers. The planner wraps non-MoE families as opaque nodes so an interleaved scheduler can still operate without lying about per-layer capabilities. It also contains special handling for NAM56R MoE-only layers where attention is effectively an identity path and the layer is mostly about expert execution. That is a concrete sign that the schedule must know not only that EP exists, but exactly where it exists.

A useful way to think about the architecture is this:

```text
Pattern: A E M E A E M E A E M R
Meaning:
A = sequence mixing via attention
E = expert or dense FFN family
M = Mamba sequence mixing
R = recurrent or M2RNN-style sequence mixing

EP target surface: only the E positions
Other families: still need their own TP/SP/CP/PP story
```

Once you frame it that way, several practical questions become clearer. Should expert weights be sharded differently from dense projections? Yes. Can a hybrid model need TP on attention while using EP on MoE? Absolutely. Can an EP win still be hidden by unrelated costs in `A` or `R` families? Also yes.

## CUDA process groups versus XLA sharding

One of the most useful details in the main model runtime module is easy to miss: CUDA and XLA do not express expert distribution the same way. The config comments note that when `--expert_parallel > 1` is used on CUDA, the runtime sets process-group-backed EP state. On XLA, the equivalent lane stays with mesh sharding over full-sized tensors instead of explicit process groups.

That distinction matters for two reasons. First, it changes what “supported EP” means operationally. A CUDA lane may be bottlenecked by expert exchange, overlap policy, or backend choice between native and Megatron-like paths. An XLA lane may instead be constrained by sharding annotations, compile stability, or how cleanly the optimizer and reductions stay shape-stable. Second, it prevents incorrect apples-to-apples comparisons between GPU and TPU MoE behavior.

The reports in the POC show exactly why this nuance matters. The TPU bug pass calls out silent fallback risks and XLA-specific reduction behavior. The compile warmup note isolates `regional_compile + MoE` as its own regression category on H200. These are not the same failure modes, even though both lanes can honestly claim to “run MoE.”

That is why high-quality writeups should say which substrate they mean:

| Substrate | EP expression | Likely operational concern |
| --- | --- | --- |
| CUDA | explicit process groups and dispatch overlap | exchange cost, compile warmup, kernel/backend choice |
| XLA / TPU | mesh sharding and compiled reductions | shape stability, fallback behavior, compile amortization |

Without that distinction, performance and correctness reports become too vague to act on.

## Shared experts, aux loss, and the limits of a sharding-only story

The POC also exposes two concepts that are often hand-waved away in high-level MoE explainers: shared experts and router-side losses. `moe_n_shared_experts`, `moe_shared_expert_gate`, and `moe_shared_expert_overlap` are all first-class config fields. That means the runtime recognizes the always-on shared path as a material part of execution, not a tiny embellishment.

This matters because shared experts complicate the clean “send tokens to one owner” picture. Some computation stays global or always active. Some can overlap with routed dispatch on CUDA, but the code comments clearly say that XLA, CPU, and compiled lanes may fall back to a sequential path. So even before you discuss aux loss, you already have a richer execution graph than the toy EP mental model suggests.

The loss terms deepen that point. `moe_aux_loss_weight` and `moe_router_z_loss_weight` are not just scalar decorations. They encode load-balancing and router regularization assumptions that have to survive parallel execution. If pipeline stages reconstruct losses incorrectly, or if resume/checkpoint paths ignore supporting modules, the run may keep stepping while drifting from the intended objective.

This is one reason the strict bug review remains relevant to an EP article. Its checkpointing warning is a reminder that distributed training correctness is broader than forward dispatch. If out-of-model training modules are not restored on resume, an MoE lane can be numerically coherent per step but still be operationally wrong over time.

## What a grounded MoE launch actually looks like

The recipes and config surfaces imply a launch style that is more explicit than many blog posts admit. A representative setup needs to say what architecture pattern is being built, how many routed experts exist, how many are active per token, whether shared experts are enabled, and how the expert path is partitioned relative to tensor parallel surfaces.

```text
example expert-parallel recipe:
  pattern = AEMEAEMEAEMR
  moe_enabled = true
  routed_experts = 16
  top_k = 4
  expert_hidden = 896
  shared_expert_hidden = 1024
  expert_parallel = 2
  expert_tensor_parallel = 1
```

The exact launcher differs by environment, but the grounded point is stable: the MoE story is inseparable from the architecture pattern and from the dense-parallel choices around it. In MegaCpp’s NAM56R recipe, those dimensions are part of the definition, not optional afterthoughts.

That is also why the phrase “capacity is cheap” needs qualification. Parameter capacity is comparatively cheap once experts are partitioned. But routing, dispatch overlap, scheduler awareness, warmup policy, and loss integrity are not cheap. They are the real system.

## What should be carried forward

A good EP mental model for this stack has five parts. First, expert ownership attaches specifically to `E` surfaces inside a hybrid pattern such as NAM56R, not to the whole model. Second, router semantics and dispatch backends are first-order runtime behavior, not tuning garnish. Third, CUDA and XLA express EP differently, so correctness and performance claims must name the substrate. Fourth, shared experts and aux or z-loss terms are part of the MoE contract. Fifth, schedule planners need architecture-aware layer typing, which is why pattern notation like `AEMEAEMEAEMR` is valuable rather than decorative.

If you preserve those five points, you avoid most of the common misunderstandings. You stop treating EP as a magical global compression trick and start treating it as a precise distributed contract around `eblock` surfaces. That is exactly how the code treats it, and that is why the code is the right source of truth.

## References

- the main model runtime module
- sanitized MoE tests
- `hybrid_schedule_plan.py`
- `nam56r_nemo_recipe.py`
- `current_live_bugs_2026-03-07_strict_pass3.md`
- `h200_compile_warmup_regression_2026-03-28.md`
