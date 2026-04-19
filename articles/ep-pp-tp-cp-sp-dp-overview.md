---
title: "EP, PP, TP, CP, SP, DP: The Parallelism Map We Actually Use"
date: 2026-04-18
author: Engineering Team
tags: [distributed-training, expert-parallel, pipeline-parallel, tensor-parallel]
summary: >
  A practical overview of the main parallelism vocabulary in modern LLM
  training, grounded in public framework semantics rather than vendor diagrams.
description: >
  What data, tensor, sequence, context, pipeline, and expert parallelism each
  own, how they compose, and where the real integration risks still live.
---

# EP, PP, TP, CP, SP, DP: The Parallelism Map We Actually Use

These acronyms are not six knobs that all do “more scale.” They each own a different failure mode. DP owns replication and optimizer state economics. TP owns large matrix partitions. SP keeps activation layout compatible with TP. CP makes long context workable when sequence length itself is the problem. EP owns MoE capacity and token routing. PP owns stage boundaries, schedule semantics, and loss accounting. The system gets healthier when each boundary has one clear owner, and it gets fragile when two modes both think they own the same tensor or loss path.

The useful way to understand distributed training is not to memorize definitions in isolation. It is to ask what resource pressure each mode is relieving and what new contract it introduces. Public bring-up notes are unusually clear on that point because the evidence is spread across launcher arguments, schedule code, and verification reports rather than hidden in marketing diagrams. The best receipts do not say “parallelism is supported.” They say exactly which composition is alive, where the next blocker moved, and which semantic edge still needs explicit wiring.

## One table that matches the code

| Mode | What it partitions | Primary benefit | Contract it introduces |
| --- | --- | --- | --- |
| DP | Replicated trainers over different data shards | Larger global batch, simpler scaling | Gradients and optimizer state must reconcile cleanly |
| TP | Weight dimensions inside large projections | Memory fit and matmul scaling | Tensor layouts become sharded through the layer body |
| SP | Sequence activations across TP ranks | Lower activation residency under TP | Sequence gather/scatter has to happen at the right boundaries |
| CP | Long-context work across context ranks | Makes larger sequence lengths feasible | Attention and state-space paths need context-aware collectives |
| EP | Experts and token dispatch | MoE capacity scaling at bounded active FLOPs | Routing, combine, and load balance become first-class concerns |
| PP | Model depth across stages | Fits deeper stacks and can overlap stage work | Stage-local loss, microbatch, and group semantics must stay correct |

That table is not theory-first. The training configuration layer exposes the same separation in flags and help text, and the H200 bring-up report keeps proving that support for one composition is not evidence for another. The receipts treat `TP + SP + FSDP2 + compile`, `PP + TP + SP + compile`, and `TP + SP + EP + FSDP2 + compile` as distinct frontiers because they are distinct contracts.

The other important thing the code makes clear is that FSDP2 is not a seventh replacement acronym that subsumes the rest. It is one implementation of the DP family. That is why the strongest receipts phrase results as compositions such as `TP + SP + EP + FSDP2`, not as “FSDP2 mode.” The runtime still has to satisfy TP layout contracts, SP activation contracts, and EP routing contracts at the same time.

## TP and SP are a pair, not a coincidence

If you only keep one composition rule in your head, keep this one: TP without SP is often an incomplete story for training, especially once activation pressure matters. TP splits projections and other large tensor operations so that one rank no longer owns the whole weight or whole intermediate. That is the easy part conceptually. The harder part is what happens to the sequence-shaped activations that move between those projections.

In practice, the healthy path is to let SP carry that activation-layout burden. That is why bring-up notes often treat `TP + SP` as a natural baseline before adding more dimensions. Once TP is on, SP is no longer a random optional accelerator. It becomes the mechanism that prevents every rank from silently rematerializing full-sequence views at every boundary.

Public Megatron-style excerpts reflect the same logic in a different form. When a path opts out of Megatron SP/TP complexity in a minimal shared block, it has to do so intentionally. The default assumption elsewhere is that SP and TP shape the runtime contract together.

The right mental model is simple: TP answers “who owns this chunk of compute?” SP answers “who owns this chunk of sequence activation while that compute happens?” If a feature claims to support TP but repeatedly falls back to fully gathered sequence activations, it is not really carrying the TP memory contract all the way through.

```yaml
parallelism:
  data_parallel: fsdp2
  tensor_parallel: 2
  sequence_parallel: true
  context_parallel: 1
  expert_parallel: 1
  pipeline_parallel: 1
```

That is not a literal checked-in launcher, but it is the cleanest baseline shape implied by the receipts: establish TP plus SP first, then add new dimensions one at a time until the next honest blocker appears.

## EP is not “TP for MoE”

People often explain expert parallelism as if it were just another way to shard a big layer. That undersells the difference. TP partitions a tensor operation that already exists. EP changes the runtime into a routing problem. Tokens must be scored, assigned, moved, computed, and combined back into the model stream. That means EP owns both capacity distribution and the communication that follows from that choice.

One representative H200 receipt is blunt about this. A major blocker on the `TP + SP + EP + FSDP2` lane was not a generic compiler issue. It was wrong EP-active detection that caused the `Block+MoE` path to take a non-EP route when `expert_tp_mesh` was still `None` even though expert parallelism was logically on. That is exactly the kind of failure you get when you pretend EP is just another tensor split. It is not. It has its own mesh semantics and its own dispatch/combine ownership.

Public DSA notes show the same difference from a deployment angle. Index-cache work is not about tensor sharding in the TP sense. It is about reducing repeated indexer work across DSA layers and handling the fact that when PP splits DSA layers across stages, a shared layer may suddenly have no preceding full layer on that stage and must be auto-promoted. That is a routing and stage-boundary story, not a matrix-partition story.

For NAM56R-style lanes this distinction matters even more. Public MegaCpp samples keep pattern strings like `AEMEAEMEAEMR` grounded in launch logic, and the corresponding public docs describe MoE with routed experts, top-k routing, and shared-expert behavior. The `E` in that pattern is not just “a heavier dense layer.” It introduces a separate token ownership problem, which is why EP gets its own axis.

## PP changes semantics, not just placement

Pipeline parallelism is the easiest acronym to misuse because the headline sounds so innocent: split the model into stages. That description is technically correct and operationally insufficient. PP changes when losses are available, how microbatches move, which process groups exist, and what counts as a valid input contract for a given torch pipeline schedule.

The H200 bring-up report is the best evidence in the tree for this point. The report does not merely say that `PP + TP + SP` works. It documents failures around `_pp_group = dist.new_group(...)`, optimizer routing that forgot to divide effective DP by `pp_degree`, unbatched PP P2P warnings, and the fact that `--pp_microbatches=2` with `--device_batch_size=1` was not a valid standard-PP input contract on that torch lane. None of that is just placement. It is semantics.

That also explains why PP is where auxiliary losses and MoE accounting often drift first. Once the model is stage-cut, any side loss or extra bookkeeping path must be explicitly PP-safe. A dense CE-only path can look fine while MoE auxiliary or z-loss logic is still subtly stage-wrong. The right lesson is not “PP is brittle.” The lesson is that PP makes latent accounting mistakes visible.

```text
example distributed launch:
  tensor_parallel = 2
  sequence_parallel = true
  pipeline_parallel = 2
  pipeline_microbatches = 4
```

Again, that snippet is schematic. The important part is the shape of the contract: once PP is on, process-group math, microbatch counts, and loss routing are no longer background details.

## CP is the long-context specialist

Context parallelism gets muddled with SP because both touch sequence-shaped data, but the code and docs make a cleaner distinction. SP is primarily an activation-layout companion to TP. CP is about making the context length itself feasible when sequence length has outgrown one rank’s comfortable ownership.

The TPU and long-context notes are where this becomes legible. CP is not present as a default on every launcher because it should not be. It exists for a specific problem: when even TP plus SP is not enough to make long contexts healthy. Public documentation also hints at this by explicitly discussing CP gathers for keys and values. That is exactly the kind of boundary where CP is doing real work that SP does not replace.

The practical consequence is that CP should be thought of as a specialty mode with a high payoff on the right workloads, not as a universal elegance layer. If your model is not context-bound, CP may add complexity faster than it adds value. If your model is long-context and attention-style paths are dominated by sequence length, CP becomes one of the few honest ways to keep scaling.

## Why the block notation matters

The architecture notation is not decoration. It is a compact way to talk about where parallelism pressure comes from. In both repos, the glossary is stable enough to be useful: `A` means attention, `M` means Mamba, `E` means expert or MoE, and `R` means recurrent. Code-facing names such as `ablock`, `mblock`, `eblock`, and `rblock` mirror those families. `cblock` usually names a composite wrapper or higher-level block shell.

Pattern strings like `AEMEAEMEAEMR` matter because they tell you which parallel mode is likely to be stressed where. `A` blocks are natural TP and SP customers because of projection-heavy attention math. `E` blocks are EP territory because the problem is routing plus expert ownership. `M` and `R` blocks often push memory and scheduling in ways that interact differently with CP and PP than pure attention stacks do.

That is why naming should stay intact in receipts. If one engineer says “the hybrid lane with recurrent tail” and another says `AEMEAEMEAEMR`, those are not equally precise. The pattern string is closer to the code and therefore closer to reproducibility.

## The operational rule: one owner per boundary

The most reusable rule across all receipts is not “enable more axes.” It is “give each boundary one owner.” If TP owns projection layout, do not let some downstream helper silently rebuild a dense view early. If EP owns expert routing, do not expect DP or FSDP2 to paper over routing mistakes. If PP owns stage boundaries, do not smuggle side-loss accounting around it. If CP owns long-context partitioning for a path, make that gather/scatter explicit.

This rule sounds obvious only after a bug has been fixed. Before the fix, the failures usually present as something more mysterious: a compile blocker, a process-group mismatch, a silent throughput collapse, or a loss path that only fails on one composite lane. The H200 bring-up history repeatedly resolved those failures by clarifying ownership, not by inventing a new abstraction layer.

The good news is that this rule scales. It works across CUDA and TPU lanes. The cost is discipline in naming, receipts, and staged bring-up. The payoff is that “parallelism support” starts to mean something specific.

## Taxonomy corrections that prevent category mistakes

- SP is not a synonym for CP. SP is a TP companion that keeps activations sequence-sharded within TP-aware regions. CP is a long-context distribution strategy.
- FSDP2 is not a separate axis beside DP. It is one implementation of DP-style sharding in native PyTorch.
- DCP means distributed checkpointing, not distributed context parallelism.
- Activation recompute and checkpointing are memory-management techniques, not split axes.
- EP owns expert routing and expert residency. PP owns stage placement and schedule semantics. TP owns large tensor partitions. Mixing those ownership boundaries in prose usually hides the real failure mode.

## References

- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- https://docs.pytorch.org/docs/stable/notes/ddp.html
- https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- https://docs.pytorch.org/docs/main/distributed.checkpoint.html
