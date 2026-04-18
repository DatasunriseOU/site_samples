---
title: "SLM architecture after the prototype: hybrid patterns, block ownership, and what survived into MegaCpp"
description: "A grounded architectural read of the small-language-model stack across the prototype and MegaCpp: hybrid patterns, block semantics, schedule ownership, and why NAM56R-style notation is more than branding."
date: "2026-04-18"
tags: ["architecture", "slm", "hybrid-models", "mamba", "moe", "nam56r"]
---

TL;DR: the SLM stack in these repos is not a smaller copy of a plain transformer. It is a pattern-driven hybrid system where `A`, `M`, `E`, and `R` blocks carry different semantics, runtime costs, and scheduling rules. The design docs in the prototype and the restoration work in MegaCpp both point to the same conclusion: if you want a believable NAM52 or NAM56R-class architecture, you have to make pattern ownership explicit in the recipe layer, the runtime schedule, and the verification surface.

## Why pattern notation matters

The architecture docs in the research repo do not treat pattern strings like `AEMEAEMEAEMR` as cosmetic shorthand. They use them to describe a real division of labor between block families. `A` stands for attention, `M` for Mamba-style state-space blocks, `E` for expert or MoE blocks, and `R` for a recurrent-style tail. That notation became especially important once the project moved away from “every layer is a transformer layer with optional extras” toward a more explicit hybrid model.

This is visible in several places. a model architecture design note, `v4_architecture.md`, and `architecture_and_eval_ru.md` all frame the model as a composite system rather than a monolith. On the MegaCpp side, the public NAM56R recipe sample, the public NAM56R Megatron recipe sample, and the public hybrid schedule sample make the same idea executable.

Once you accept the notation as real architecture, several downstream choices become easier to reason about:

- the recipe layer can build pattern-aware layer specs;
- the scheduler can decide whether a layer is transformer-like, opaque, or MoE-specialized;
- memory and compile discussions can talk about the actual source of cost rather than a uniform “layer” fiction;
- verification can check that the implemented pattern matches the declared pattern.

That last point matters more than it sounds. If the repo says the model is `AEMEAEMEAEMR` but the runtime schedule still treats all layers as if they were the same class, then the pattern string is only documentation theater.

## What the prototype architecture actually emphasized

The prototype’s architecture notes pushed hard on the idea that not all capacity should live inside attention. This is where the small-model story becomes interesting. A small language model cannot simply shrink a giant transformer and hope the same tradeoffs work. It has to allocate capacity more deliberately.

The model docs emphasize several recurring themes:

- attention remains critical for token mixing and retrieval-like behavior;
- Mamba-style paths can carry temporal/stateful structure more efficiently in some regimes;
- expert layers provide bursty capacity without turning every token into a dense full-model pass;
- recurrent or specialized tail blocks can consolidate state differently than another generic attention block.

That is already a meaningful departure from a plain GPT stack. The point is not novelty for its own sake. The point is to redistribute modeling work so the small model does not waste all of its parameter budget on repeating the same mechanism everywhere.

A useful architecture summary table is:

| Block | Architectural role | Typical strength | Typical risk |
| --- | --- | --- | --- |
| `A` | global token interaction | robust general-purpose mixing | expensive if overused |
| `M` | state-space / temporal processing | efficient long-range dynamics | implementation complexity |
| `E` | conditional capacity | large effective capacity without always paying dense cost | routing and runtime complexity |
| `R` | recurrent-style consolidation | can specialize end-of-pattern behavior | easiest place to accumulate custom debt |

That table is close to how the repos actually behave. The strength of the notation is that it makes these distinctions visible early instead of burying them inside layer constructors.

## What MegaCpp restored instead of reinventing

The MegaCpp side did not just copy the architecture writeups. It translated them into recipe and runtime surfaces. a Megatron restoration note and the public changelog are especially useful because they describe restoration as a sequence of concrete compatibility decisions rather than a single “ported architecture” claim.

the public NAM56R recipe sample is one of the clearest examples. The tests around it, especially sanitized recipe regression tests, verify that a call to `build_nemo_hybrid_pattern(pattern="AEMEAEMEAEMR", depth=52, use_moe=True)` yields the expected shape. That is not just syntax. It is an architectural invariant carried into code.

the public NAM56R Megatron recipe sample then turns the same high-level pattern into a Megatron-facing configuration surface. This matters because pattern notation only becomes operational once it affects the actual constructed stack.

The most interesting runtime piece is the public hybrid schedule sample. The docstring says it plainly: mixed Mamba plus transformer layers need a schedule plan that dispatches each layer either to `TransformerLayerSchedulePlan` or to `OpaqueLayerSchedulePlan`. The file also documents a very specific NAM56R reality: MoE-only layers can have `IdentityOp` attention, so some upstream state attributes still need to exist even if those layers do not consume them in the normal transformer way.

That is the kind of detail that proves the architecture is real. Once the runtime needs special handling for MoE-only layers and non-transformer opaque layers, the pattern is no longer a slogan.

## Why small models need this level of explicitness

A small model has less room for accidental inefficiency. In a giant model, you can sometimes hide architectural redundancy behind scale. In an SLM, every repeated mechanism competes harder for the same parameter and activation budget.

This is where NAM52 and NAM56R-style accounting matters. The repo materials repeatedly separate total parameter count from active capacity, especially when MoE is involved. The H200 launcher notes and recipe tests on both sides show why: once you mix routed experts, shared experts, attention, and Mamba-style paths, “the model size” stops being a single obvious number.

The right architecture conversation therefore has at least three layers:

1. What is the declared pattern and depth?
2. What is the active path per token?
3. What runtime schedule and memory policy does that imply?

That is much more useful than saying “we built a small hybrid model.”

The hybrid schedule file supports this interpretation. It creates different plan objects based on whether the current layer is a true transformer MoE layer or an opaque layer. That means runtime already recognizes that an `E` block and an `M` block are not interchangeable just because they occupy one depth slot each.

## What survived contact with runtime and eval

The prototype architecture documents were ambitious, but not every idea survived unchanged. The changelog and report materials show that some general claims had to be narrowed once training, compile policy, and evaluation got real.

A few durable lessons stand out.

First, architecture claims need runtime receipts. The H200 compile warmup report and the live bug reviews show that some broad claims about compile behavior or sparse-path smoothness were too optimistic until the exact lane was proven.

Second, eval results have to be interpreted through the architecture lens. a checkpoint evaluation report makes it clear that raw benchmark tables can hide format mismatch, checkpoint timing problems, or capability concentrated in a subset of tasks. That is especially relevant for hybrid SLMs because a new block family may first change output structure or training dynamics before it cleanly improves benchmark pass rates.

Third, restoration into MegaCpp favored explicit surfaces over hidden magic. Recipes, tests, schedule plans, and changelog notes do more work than aspirational prose. That is the right bias for an architecture this mixed.

A practical “what survived” table looks like this:

| Architectural idea | Status after implementation | Evidence |
| --- | --- | --- |
| pattern-driven hybrid stack | kept | recipe builders and tests use `AEMEAEMEAEMR` directly |
| MoE as first-class conditional capacity | kept with runtime specialization | hybrid schedule recognizes MoE-only layer handling |
| Mamba paths as ordinary transformer layers | rejected | opaque schedule path exists for non-transformer layers |
| one-number model-size story | rejected | recipe and launcher materials separate total and active capacity |

## How to reason about ablock, mblock, eblock, and rblock

The glossary framing is still useful if it is tied to code rather than used as branding. Thinking in terms of `ablock`, `mblock`, `eblock`, and `rblock` helps because those names force you to ask which subsystem owns the block.

- `ablock` ownership is usually attention kernels, projection geometry, rope or positional surfaces, and their compile policy.
- `mblock` ownership is state-space mixers, recurrent state handling, and custom kernel/runtime concerns.
- `eblock` ownership is routing, expert parameterization, and MoE-aware scheduling.
- `rblock` ownership is the most repo-specific and therefore the one that needs the strongest documentation and tests.

That ownership model is more practical than asking whether the architecture is “mostly transformer” or “mostly Mamba.” Hybrid SLMs are not a purity contest. They are an allocation problem.

```text
Pattern example: A E M E A E M E A E M R
Interpretation:
- three attention anchors for broad token interaction
- repeated expert capacity injections
- repeated state-space processing stages
- specialized recurrent-style tail for final consolidation
```

Once the pattern is written that way, the code structure in the recipe and schedule layers becomes much easier to review.

## The architecture discipline MegaCpp should keep

The strongest lesson across both repos is that architecture has to remain executable. Pattern strings should be checked by tests. Runtime schedule should expose block ownership. Changelog entries should record where a broad claim was narrowed by real execution evidence.

If MegaCpp keeps that discipline, the SLM story stays coherent:

- pattern notation remains meaningful;
- runtime behavior stays inspectable;
- parameter accounting remains honest;
- future restorations into other backends do not have to rediscover what the blocks meant.

That is the difference between a hybrid architecture and a bag of features.


## Pattern-aware scheduling is what prevents architecture drift

The strongest MegaCpp contribution to the architecture story is that it makes drift visible. the public hybrid schedule sample does not let the runtime hide behind a generic “layer list” abstraction. It has to choose how to schedule each layer. That choice turns architecture into runtime behavior.

This matters because hybrid stacks tend to drift silently if the schedule layer is too generic. A project may still advertise an `AEMEAEMEAEMR` pattern while gradually routing most of the stack through transformer-default code paths. Once that happens, the conceptual architecture and the executed architecture are no longer the same thing.

The schedule plan reduces that risk in two ways. First, it distinguishes MoE-capable transformer layers from opaque non-transformer layers. Second, it carries enough state for upstream interfaces that still expect transformer-like attributes even when a specific layer family does not use them in the same way. The comment about NAM56R expert layers with `IdentityOp` attention is exactly this kind of bridge logic. It is not pretty, but it is honest about the mixed system the runtime is serving.

A small-model architecture needs that honesty because every hidden compatibility bridge eventually becomes architecture debt. If the bridge is explicit, it can be tested and eventually simplified. If it is hidden, the team starts losing track of which blocks actually own which semantics.

## Why evaluation language should stay pattern-aware too

One subtle but important lesson from a checkpoint evaluation report is that evaluation language should not erase architecture. When a hybrid model underperforms on a checkpoint, the question is not only “did the metric go down?” The question is also whether a particular architectural mix introduced a training or formatting regime that the benchmark is reading in a distorted way.

That matters for SLMs because the same parameter budget can be arranged in very different patterns. A model with more `E` capacity and less dense attention may train and emit outputs differently from a more conventional stack even before its final performance stabilizes. A verifier-first or report-rich evaluation style is therefore complementary to pattern-aware architecture: both refuse to compress unlike things into one generic label.

## References

- a model architecture design note
- `v4_architecture.md`
- `architecture_and_eval_ru.md`
- a checkpoint evaluation report
- a Megatron restoration note
- the public changelog
- the public NAM56R recipe sample
- the public NAM56R Megatron recipe sample
- the public hybrid schedule sample
- sanitized recipe regression tests
