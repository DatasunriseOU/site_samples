---
title: "Migration policy: native Megatron vs narrow custom seams"
description: "Why MegaCpp ports only what Megatron or Nemotron do not already provide, and why ambiguous mappings should fail closed instead of being reinterpreted silently."
date: "2026-04-19"
tags: ["migration", "megatron", "nemotron", "porting-policy"]
---

The easiest way to make a migration story unreadable is to port everything. A
clean migration policy does the opposite. Reuse native Megatron or Nemotron
surfaces where they are real, and keep only the irreducible local seams custom.

MegaCpp's migration policy is useful because it states that boundary directly.
It prefers translation layers, fail-closed mappings, and narrow local seams
instead of one large downstream fork.

## What the policy is actually buying

This is not only a code-organization preference. It makes the stack easier to
verify and easier to explain publicly.

- native surfaces stay close to upstream docs and runtime behavior
- custom seams remain enumerated and auditable
- ambiguous mappings stop early instead of silently drifting

That is why the translator, recipe, MLA adapter, and recurrent mixer examples
belong together. They are all examples of the same rule: keep the custom seam
as small as possible and make it obvious where it begins.

## References

- [Fail-closed pattern translation sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/fail_closed_pattern_translation_sample.py)
- [Nemotron recipe to Megatron sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nemotron_recipe_to_megatron_sample.py)
- [MLA shared adapter sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mla_shared_adapter_sample.py)
- [M2RNN mixer spec sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/m2rnn_mixer_spec_sample.py)
- [Migration matrix](https://github.com/DatasunriseOU/cppmega/blob/main/docs/migration_matrix.md)
- [Porting policy](https://github.com/DatasunriseOU/cppmega/blob/main/docs/porting_policy.md)
