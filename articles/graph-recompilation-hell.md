---
title: "Graph Recompilation Hell: How Small Runtime Choices Turned Into Full TPU Rebuilds"
description: "What repeatedly triggered recompilation on the TPU lane, how the project narrowed the causes, and which coding rules came out of that work."
date: "2026-04-18"
tags: ["xla", "compile", "recompile", "tpu", "torch-compile"]
summary: >
  The expensive part was not one mythical compiler bug. It was a pile of small
  runtime choices that kept invalidating graph assumptions across the TPU lane.
---

# Graph Recompilation Hell: How Small Runtime Choices Turned Into Full TPU Rebuilds

**TL;DR:** the project did not escape recompilation pain by finding one magic flag. It got there by separating true compiler blockers from self-inflicted graph invalidations: feature toggles with unstable shapes, mode switches that changed execution structure, and MoE or sparse paths whose compile story differed sharply from their eager story. The reliable lesson is to freeze runtime structure first, then reintroduce complexity rung by rung.

Short summaries of compiler trouble tend to blame "XLA being weird" or "torch.compile being unstable." That is not wrong often enough to be useful. The source tree tells a more actionable story. In the TPU lane, recompilation pressure came from repeated changes to graph structure, tensor shapes, and backend choice. In the H200 receipts and broader changelog, a similar pattern appears in CUDA form: compile-safe dense lanes and compile-hostile MoE or sparse corners are not the same problem, and they should not be debugged as one blob.

The useful mental model is this: recompilation hell is a systems problem created by unstable contracts. Every time the runtime changes a shape, branch family, or backend expectation that the compiled graph treated as fixed, the cost is not just a local slowdown. The cost is another graph specialization, another compile miss, or another fallback path that confuses throughput receipts.

## Start by Separating Structural Drift From Real Compiler Defects

One of the strongest clues in the repo is the repeated use of validation ladders and explicit receipts instead of broad claims. the TPU feature-ladder validation flow is small on purpose. It starts from a constrained canary and adds features in a controlled order: NCP, then Mamba, then MoE, then modulation, Engram, MHC, DSA, and later MTP-like features. That script embodies the right debugging rule. If the graph keeps recompiling, do not begin with the full hybrid stack. Begin with the smallest lane whose runtime structure you can actually hold still.

This same discipline shows up on the GPU side in an H200 bring-up receipt. That report distinguishes between a dense TP+SP+FSDP compile lane that passes and a later real MoE frontier that fails inside standalone `TokenChoiceMoELayer`. That matters because it disproves the lazy explanation that "compile is broken everywhere." It is not. One lane is stable; another lane crosses a different structural boundary.

That is the right standard for recompilation analysis. Once one narrow lane is known-good, every later failure should be described in terms of the structural feature added on top of it. Otherwise the investigation keeps forgetting what is already proven.

| Problem shape | What it looks like | Correct response |
| --- | --- | --- |
| Graph-structure drift | Different branches, feature flags, or layer families per step | Freeze runtime mode and re-test |
| Shape drift | Different token counts, capacities, or batch geometry | Pin shapes or constrain ladder input |
| Backend mismatch | Dense path compiles, sparse or jagged path does not | Split receipts by backend and family |
| Genuine compiler failure | Minimal frozen repro still recompiles or crashes | Escalate with a minimal reproducer |

Without this separation, every throughput dip turns into compiler folklore.

## MoE and Sparse Paths Were Not Compile-Equivalent to Dense Paths

The changelog is unusually explicit here. Multiple entries document that jagged grouped MoE paths and certain sparse attention or specialized kernels had a very different compile story from the padded or dense alternatives. `CHANGELOG.md` describes a concrete issue where jagged `grouped_mm` behavior broke compile tracing, leading to graph breaks or forcing `@torch.compiler.disable` on fused MoE segments. It also records the practical conclusion: in some configurations the padded path, despite nominal waste, was faster overall because it preserved a more compilable graph.

That is a harsh but useful lesson. The fastest eager kernel is not automatically the fastest compiled training lane. If a specialized operator shatters the graph, the overhead can dominate the local arithmetic win.

The same theme appears in the March sparse-attention recovery notes. a DSA/SDPA recovery changelog separates a correctness-preserving sparse SDPA recovery from a larger throughput regression story. That note exists because the team learned not to mix "we restored correct backend behavior" with "all performance regressions are now explained." Recompilation debugging requires that same honesty.

```text
example TPU validation ladder:
  run base configuration
  run mamba-enabled configuration
  run mamba-plus-moe configuration
  compare cache stability, recompilation count, and throughput after each rung
```

This ladder is not just a TPU convenience script. It is a way to ask, with evidence, where compile structure stops being stable.

## Runtime Mode Drift Was Often Self-Inflicted

A recurring theme in the repo is that configuration validation matters as much as operator quality. the training-arguments layer and the main training entrypoint are not glamorous files, but they are where unstable mode combinations get either stopped early or allowed to poison a run. When runtime mode changes are accepted too late, the project pays twice: first in invalid user expectations, then in opaque compile behavior.

The completion plan and related tests reinforce this point. There are multiple notes about moving validation earlier so incompatible settings fail in shared argument handling instead of surfacing deep in execution. That is not mere UX polish. Early rejection reduces graph drift because the program is less likely to build different internal structures depending on a combination that should have been illegal from the start.

This is especially relevant in hybrid stacks. NAM52 and NAM56R-style lanes combine dense attention, Mamba-style blocks, MoE, optional sparse attention variants, and auxiliary subsystems. If the runtime flips between these structural modes across runs or even across steps, recompilation is not a surprise. It is the expected outcome of letting architecture-level decisions behave like dynamic control flow.

## The Prototype Learned to Treat Receipts as Compile Boundaries

The H200 bring-up report is nominally about another platform, but its lesson applies directly. The report repeatedly narrows failures to a specific lane with a specific topology and a specific operator family. Dense TP+SP+FSDP compile is marked alive. A later standalone MoE unit is marked the next real blocker. The report also records when a local input seam or `DeviceCopy in input program` warning disappeared and when it reappeared. That is what disciplined compile work looks like: not "compile is good now," but "this exact structural boundary is now stable, and that other one is not."

That style is worth copying because it keeps progress and uncertainty on the same page. A team can celebrate a passing compile lane without accidentally asserting that every adjacent family is now safe. Receipts become directional, not absolute.

On TPU, the same style should be the default. A receipt should name the exact feature rung, shape envelope, and backend family. Otherwise engineers start generalizing from the wrong result. If `base_ncp` is stable, that does not prove `ncp_mamba_moe_mod_engram_mhc_dsa` is stable. If a DSA recovery restores throughput for one sparse path, that does not prove every sparse backend is compile-friendly.

| Receipt style | Bad version | Good version |
| --- | --- | --- |
| Compiler status | "compile works" | "compile stable on base NCP canary at fixed depth and seq len" |
| Feature claim | "MoE is fine" | "padded MoE lane stable; jagged grouped path still graph-break prone" |
| Sparse claim | "attention fixed" | "SDPA recovery correct; broader sparse backend regression remains separate" |
| Hybrid claim | "NAM56R works" | "specific rung and block mix stable under fixed runtime contract" |

This precision is boring, but it is how teams stop re-learning the same failure.

## Why TPU Recompilation Felt Worse Than It Was

Recompilation bugs often feel mystical because the visible symptom is delayed. The run launches, some early steps look fine, and then compile caches miss or backend structure changes. On TPU this feels especially painful because compilation cost is large enough to distort every nearby metric. Tokens per second, startup time, and even qualitative debugging intuition all become unreliable once the graph is churning.

That is why the operator notes around TPU v6e and related runtime bundles emphasize stable bring-up and preserved runtime evidence rather than sweeping claims. The project had to learn that platform notes, feature-ladder outcomes, and backend recovery receipts are all different evidence types. Mixing them together turns the compile story into noise.

Another reason it felt worse is that some fixes were real but narrow. A sparse SDPA recovery helped. Earlier validation of invalid mode combinations helped. Dense compile receipts on other platforms proved that not every compile complaint was fundamental. But each of those wins only removed one class of instability. They did not erase the cost of dynamic shapes, heterogeneous block families, or compile-hostile jagged operators.

That narrowness is important. In compiler bring-up, one fix often just reveals the next blocker. The mistake is interpreting that reveal as proof that the previous blocker was imaginary. The repo's better receipts avoid that trap by preserving the staircase explicitly.

## The Coding Rules That Actually Helped

The code and docs point to a small number of durable rules.

First, keep runtime structure fixed while validating compile. That means fixed depth, fixed sequence length, fixed batch geometry, fixed feature rung, and no opportunistic backend swaps hidden behind one broad recipe.

Second, split dense, sparse, and MoE receipts. The changelog shows clearly that these lanes can have different compile behavior even when they belong to the same model family.

Third, move incompatibility checks as early as possible. If `train_args` can reject a structurally invalid combination, it should. Late rejection is a compile tax.

Fourth, name the real frontier. If the current blocker is standalone `TokenChoiceMoELayer`, say that. If the current stable rung is `base_ncp`, say that. Engineers only stop thrashing when the failure surface is narrow enough to own.

Finally, treat pattern notation like `AEMEAEMEAEMR` as a compile clue, not just a model-description flourish. A hybrid sequence of block families is a warning that graph structure may change in ways a dense-only intuition will miss.

One more rule follows from that. Never compare compile receipts across lanes that do not share the same pattern semantics. A dense-heavy NAM52 slice and a richer NAM56R slice may share tooling, but they do not share the same graph-risk envelope. Stability in the first lane does not automatically transfer to the second.

## What to Keep in Mind for NAM52 and NAM56R Work

For NAM52-style compile bring-up, the main win is usually to establish a fully frozen dense or lightly hybrid lane and keep it honest. For NAM56R-style work, especially where the pattern mix becomes richer and long-context pressure rises, the danger is pretending one stable subset proves the whole pattern is compile-clean.

That is why the repo's best artifacts are receipts and ladders, not slogans. They show where the graph is stable, where it is not, and which runtime choices are responsible.

Graph recompilation hell was never one monster. It was a pile of small freedoms the runtime should not have taken during compile validation. Once those freedoms were constrained, the remaining compiler bugs became much easier to see.

That is the real takeaway from the code-backed evidence. Stable compile behavior is usually won by reducing the number of ways the runtime can surprise the compiler. Only after that discipline is in place do the remaining backend and lowering bugs become worth treating as the main story.

## References

- the TPU feature-ladder validation flow
- the main training entrypoint
- the training-arguments layer
- an H200 bring-up receipt
- a DSA/SDPA recovery changelog
- `CHANGELOG.md`
