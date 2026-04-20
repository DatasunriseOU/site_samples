---
title: "How to express a Nemotron-style recipe as pure Megatron CLI"
description: "Why MegaCpp keeps high-level recipe objects and then lowers them into a smaller native Megatron flag surface instead of treating one giant launcher as the source of truth."
date: "2026-04-19"
tags: ["megatron", "nemotron", "recipes", "launchers", "nam56r"]
---

The useful question is not whether a model can be launched from a shell script.
Almost anything can. The useful question is whether the launch surface still
distinguishes native Megatron flags from custom model seams once the recipe gets
large.

MegaCpp's answer is to keep a recipe-level authoring surface and then lower it
into a smaller native Megatron CLI bundle. That is what makes the launch
contract readable. The recipe carries pattern, topology, and feature intent.
The emitted CLI carries the part Megatron can actually own.

## Why a recipe object is better than a shell blob

Hybrid models accumulate too many interacting switches to keep the launcher
string as the only source of truth. Pattern layout, Mamba or recurrent seams,
MLA, MTP, CUDA graph policy, and MoE settings are not all the same kind of
decision. Treating them as one flat shell string is how teams lose track of
which part is native and which part is local.

The public MegaCpp examples keep that separation visible:

- a Nemotron-style recipe sample holds the full authoring intent
- a Megatron-args sample emits only the native runtime flags
- a launcher sample shows how the two are combined without pretending they are
  the same layer of abstraction

That is the right public contract. Native flags should stay native. Custom
seams should stay explicit.

## What gets lowered and what does not

The practical lowering rule is simple.

Megatron-native runtime concerns such as tensor parallel size, pipeline
parallel size, sequence parallel, and other well-supported execution flags can
be emitted directly. Custom hybrid surfaces such as unsupported block families,
custom recurrent seams, or special launch-time graph policy should remain as
recipe-level notes or separate launch helpers.

That is why a good recipe translator is not only an arg emitter. It is also a
boundary keeper.

## Where this lands in MegaCpp

MegaCpp keeps this flow as a stack of explicit surfaces:

- recipe intent
- fail-closed pattern translation
- Megatron-native arg emission
- final launch helper

That stack is more useful than one long launcher because it makes migration and
debugging cheaper. If a feature cannot lower cleanly into native Megatron, the
translator should say so instead of silently approximating it.

## References

- [Nemotron recipe to Megatron sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nemotron_recipe_to_megatron_sample.py)
- [Megatron args sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/megatron_args_sample.py)
- [NAM56R launcher profile sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_launcher_profile_sample.py)
- [Migration matrix](https://github.com/DatasunriseOU/cppmega/blob/main/docs/migration_matrix.md)
- [Megatron Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
