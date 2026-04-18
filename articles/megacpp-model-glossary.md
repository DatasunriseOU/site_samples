---
title: "MegaCpp model glossary: patterns, blocks, and what names like NAM52 and NAM56R actually encode"
description: "A grounded glossary for MegaCpp model notation, hybrid layer patterns, and block-family names, tied back to live builder code, launch helpers, and regression tests in MegaCpp and the research repo."
date: "2026-04-18"
tags: ["glossary", "models", "mamba", "moe", "attention", "architecture"]
---

**TL;DR:** this naming scheme is an execution vocabulary, not branding. In this stack, `A`, `M`, `E`, and `R` are stable symbols for attention, Mamba-family state-space layers, expert/MoE layers, and recurrent tails. Strings such as `AEMEAEMEAEMR` are used by launch builders and tests to describe ordered hybrid layouts. Names such as `NAM52` and `NAM56R` are shorthand for concrete recipes whose real meaning lives in builder code, schedule patches, precision gates, and launch assertions rather than in any single README.

When people first meet this codebase, the model names look denser than they really are. The confusion comes from seeing a launch string, a report, and a unit test each preserving a different slice of the same contract. The way to read the notation correctly is to start from the code paths that consume it. In sanitized Megatron-args tests, the feature-plan builder is called with `pattern="AEMEAEMEAEMR"` and `depth=52`, then checked for downstream flags such as MLA, MTP, FIM, MoE, and DSA. In sanitized NAM56R launch tests, the same pattern is expanded into the launch notation used by the Megatron-side recipe. In the public selective-FP8 MoE patch sample, the docstring explains why only the expert layers run in FP8 for the `NAM56R` family. Those files together tell you that the notation is operational: it decides how the model is assembled, scheduled, and optimized.

## The stable symbols: A, M, E, and R

The safest part of the glossary is the four-letter alphabet. It is repeated in tests, builder helpers, and patch docstrings, and it is the minimum information you need before reading any pattern string.

| Symbol | Meaning in this stack | Where the meaning is grounded |
| --- | --- | --- |
| `A` | attention layer | launch-pattern tests and selective precision docs |
| `M` | Mamba-family state-space layer | the public TE mixer sample, the public TE stack spec sample |
| `E` | expert / MoE layer | selective FP8 MoE path, hybrid schedule logic |
| `R` | recurrent tail or custom recurrent-style layer | sanitized NAM56R launch tests custom-layer indices |

That mapping is not inferred from blog prose. The clearest compact summary is the module docstring in the public selective-FP8 MoE patch sample, which says that the `NAM56R` family uses the pattern `AEMEAEMEAEMR` and explicitly glosses `A=attention`, `E=MoE`, `M=Mamba/Mamba3`, and `R=M2RNN`. The launch tests then prove that the pattern is not just documentation. `get_custom_layer_indices(pattern="AEMEAEMEAEMR", depth=52, custom_symbols=("R",))` is expected to return `(12, 24, 36, 48)`, which means the recurrent symbol is discovered programmatically after tiling the pattern through depth.

That is the first useful mental model: the symbols are the architectural primitive types, and the pattern string is the ordered list from which deeper launch syntax is derived.

## What pattern strings actually do

A string like `AEMEAEMEAEMR` is not a label pasted on the side of the model after the fact. It is a compact serialization of layer order. That matters because this repo does not treat the stack as one repeated homogeneous block. It composes multiple families and then feeds that composition into launch helpers.

The pattern-expansion behavior is easiest to see in sanitized NAM56R launch tests. One test calls `build_nam56r_lite_main_pattern(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1)` and asserts that the output contains 17 `M` symbols, 22 `E` symbols, and 14 attention placeholders. Another test flips `use_dsa_symbol=True` and verifies that all attention positions become `D`, producing 14 D symbols and an ending of `/D-`. That tells you two important things.

First, the repeated source pattern is a template that is expanded across model depth. Second, the expanded form can be rewritten for downstream execution modes. Attention is not always emitted as plain attention in the launch string; when the DSA path is enabled, the symbolic schedule reflects that. In other words, the notation is not frozen. It is a boundary representation between architectural intent and runtime specialization.

A small example helps:

```text
Base reference pattern:   A E M E A E M E A E M R
Tiled through depth:      ... repeated to 52 layers ...
Launch-facing rewrite:    attention may become '*' or 'D'
Custom layer extraction:  R positions become tracked tail indices
```

This is why you should resist the temptation to read a single literal string as the whole truth. The authoritative reading is “pattern plus expansion rules plus feature toggles.”

## NAM52 and NAM56R are recipe handles, not universal standards

The names `NAM52` and `NAM56R` look like official external model families, but inside this project they behave more like recipe handles. They point to a bundle of assumptions: depth, hybrid layout, enabled features, and sometimes hardware-facing expectations captured in notes or patch docstrings.

`NAM56R` is the more explicit one in the current tree. It appears in launch tests, memory-budget notes, FP8 patches, and Mamba modules. the public linear-CE patch sample mentions memory savings at `NAM56R` batch sizes. the public Hopper-native CE sample contains an explicit note about memory cost at `NAM56R MBS=10`. the public PSIV cache sample talks about staying under the remaining GPU budget at `NAM56R MBS=8`. Those references are not there because the name is fashionable. They are there because the recipe is used as a calibration target for throughput and memory work.

`NAM52` shows up more in reports, for example in the prefill receipt under a NAM52 a modern accelerator prefill receipt. In practice that means `NAM52` and `NAM56R` are best read as stable checkpoints in the project’s experimentation history. They are useful because different patches, reports, and launch helpers agree on them. They are risky when treated as self-explanatory. If you need exact semantics, you still have to read the launch recipe or the feature plan that is active in that lane.

| Name | Safe interpretation | What not to assume |
| --- | --- | --- |
| `NAM52` | a specific local recipe / benchmark family used in reports and bring-up | not a universal external architecture standard |
| `NAM56R` | a local hybrid recipe with a well-used pattern and recurrent tail notation | not a guarantee that every run with the label has identical features |

The practical rule is simple: use the names as anchors for discussions, but resolve the actual behavior from the builder path and the launch flags.

## What M actually means here: not generic Mamba, but a specific author path

The `M` symbol is where generic vocabulary becomes dangerously imprecise. In this repo, “Mamba” could mean the general state-space family, the Transformer Engine stack wrapper, or the authored Mamba3 kernel path. Those are related, but they are not interchangeable.

The top-level description in the public Mamba TE mixer sample is the most direct source. The file defines `CppMegaMamba3TE` as a drop-in replacement for `MambaMixer` that keeps TE projection layers while replacing the upstream convolution-plus-SSD scan path with authored Mamba3 kernels. The docstring names the supported behaviors explicitly: trapezoidal discretization, QK-Norm via gated RMSNorm on B/C, learnable B/C bias, complex RoPE on B/C, data-dependent `A`, and MIMO. It also states that there is no conv1d in this path; the authored scan owns the state-space computation.

the public Mamba TE stack spec sample reinforces the same picture from the stack side. Its module comment describes a stack that preserves upstream TE submodules while swapping in the authored Mamba3 mixer. So when someone casually says “this is an MBlock,” they may be referring to one of three levels:

| Informal term | Usually points to |
| --- | --- |
| `mblock` / `M` layer | a Mamba-family slot in the hybrid schedule |
| Mamba TE stack | the stack spec that preserves TE scaffolding |
| authored Mamba3 path | the specific `CppMegaMamba3TE` mixer with trapezoidal, RoPE, data-dependent `A`, and MIMO support |

That distinction matters for debugging. If a report says the Mamba lane is slow, the next question is whether the problem is in the stack integration, the authored kernel path, or an adjacent precision policy. The glossary should preserve that hierarchy instead of flattening it into “M means state space, done.”

## What E means: expert layers are runtime islands with their own precision policy

The `E` symbol is also more specific than “there is some MoE somewhere.” In the current tree, expert layers are treated as the place where certain optimizations make sense even when they do not help the rest of the model. The strongest example is the public selective-FP8 MoE patch sample.

That module explains the central claim in plain language: on the `NAM56R` family, running FP8 everywhere is slower because Mamba scans and attention remain bandwidth-bound while paying conversion overhead, but the expert FFN GEMMs inside MoE layers do benefit. The patch therefore monkey-patches Megatron’s FP8 context helper so that only the MoE layers stay in FP8 while non-MoE layers fall back to `nullcontext()` and therefore allocate in BF16. This is not a paper definition of MoE. It is a runtime definition of the expert blocks as the compute islands where reduced precision buys something measurable.

That behavior also explains why pattern notation is useful. Once you know where the `E` positions are, you can target them for feature gating, performance patches, or schedule experiments without rewriting the entire stack. The helper `_compute_moe_layer_indices()` resolves the active pattern and returns the exact zero-based `E` positions. In other words, the symbol is not just descriptive. It is how the code discovers which layers should receive a special policy.

## What R means, and why it is preserved instead of erased

The `R` symbol is easy to miss because there are fewer references to it than to `A`, `M`, or `E`. But the launch tests make clear that it is intentionally preserved long enough to drive special handling. In sanitized NAM56R launch tests, recurrent positions are extracted with `custom_symbols=("R",)`, and the expected indices are hard-coded. Another launch test checks that the final emitted pattern contains no literal `R` after the main rewrite, because the launch-facing pattern maps it into a form the downstream system understands.

That tells you the right glossary entry for `R`: it is not “a mysterious spare letter.” It marks custom recurrent-tail layers in the authoring notation, then participates in a rewrite step before the final launch string is emitted. The exact backend implementation may vary, but the presence of the symbol is a first-class part of how the recipe is specified.

This is also why local terms such as `rblock` stay useful. They give people a short name for a structural family that still needs custom handling when the launch pattern is derived.

## Where ablock, mblock, eblock, rblock, and cblock fit

The lowercase block-family words are less canonical than the four letters, but they are still useful shorthand when grounded. The project most strongly supports `ablock`, `mblock`, `eblock`, and `rblock`, because those correspond directly to the stable symbol set. Reports also use terms such as `EBlock` in exactly that spirit, for example when discussing compile behavior around MoE layers.

The safest way to use these words is as prose aliases for the symbol families:

| Term | Safe local reading |
| --- | --- |
| `ablock` | an attention-family layer or slot |
| `mblock` | a Mamba-family layer or slot |
| `eblock` | an expert/MoE layer or slot |
| `rblock` | a recurrent-tail layer or slot |
| `cblock` | only use when a nearby file or design note defines it explicitly |

That last line matters. `cblock` can be a tempting extension, but unless the immediate code or doc surface defines it, the term should be treated cautiously. The rest of the block-family words are grounded by the stable symbol map and by launch tests. `cblock` is only safe when the surrounding source defines what `C` stands for in that context.

## How to read a model name without lying to yourself

The best practical workflow is to decode names in layers.

1. Read the stable symbols first: `A`, `M`, `E`, `R`.
2. Check whether you are looking at the authoring pattern, the expanded launch pattern, or a performance note.
3. Resolve the active feature plan, because MLA, MTP, FIM, DSA, and precision patches can all change how the same structural recipe behaves.
4. Treat `NAM52` and `NAM56R` as local recipe handles whose exact meaning depends on the builder path.

A concrete snippet from the tests captures that mindset:

```python
plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1)
bundle = build_megatron_args_bundle(
    plan=plan,
    use_mla=True,
    use_mtp=True,
    use_fim=True,
    use_moe=True,
    use_dsa=True,
)
```

The recipe name alone is not enough. The effective model is the pattern plus the feature plan plus the execution policy. Once you read the notation that way, the glossary stops being mysterious. It becomes a compact map from architectural intent to runtime behavior.

## References

- the public TE mixer sample
- the public TE stack spec sample
- selective FP8 MoE implementation
- sanitized Megatron-args tests
- sanitized NAM56R launch tests
- hybrid schedule planning implementation
