---
title: "Fail-closed hybrid pattern translation"
description: "Why MegaCpp refuses to silently remap unsupported hybrid block families when translating NAM56R-style patterns into Megatron-native plans."
date: "2026-04-19"
tags: ["megatron", "hybrid-models", "pattern-translation", "nam56r"]
---

The wrong way to translate a hybrid pattern is to be helpful. Silent remapping
turns a translation layer into a source of architectural drift.

MegaCpp's safer rule is fail-closed translation. Map the supported block
families, preserve stage breaks, and stop when the pattern asks for a block the
native runtime does not actually understand.

## Why this matters

Hybrid pattern strings look compact, but they are carrying real architecture
intent. If an unsupported token is silently rewritten into something nearby,
the resulting plan may still run while no longer matching the model the recipe
author thought they described.

That is why the public translation sample is intentionally narrow. It does not
pretend every local block family has a native Megatron equivalent. It maps what
is grounded, and it stops on the rest.

## The practical benefit

Fail-closed translation makes three things cheaper:

- code review, because unsupported surfaces remain obvious
- migration, because custom seams stay enumerated instead of disappearing into
  ad hoc remaps
- article honesty, because the public docs can state exactly which pieces are
  native and which remain local

## References

- [Fail-closed pattern translation sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/fail_closed_pattern_translation_sample.py)
- [NAM56R pattern composition sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_pattern_composition_sample.py)
- [NAM56R feature placement sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_feature_placement_sample.py)
- [Migration matrix](https://github.com/DatasunriseOU/cppmega/blob/main/docs/migration_matrix.md)
