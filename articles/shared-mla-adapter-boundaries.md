---
title: "Shared MLA adapter boundaries"
description: "Why MegaCpp keeps MLA-specific normalization behind one shared adapter seam instead of leaking MLA conditionals through the whole attention builder stack."
date: "2026-04-19"
tags: ["mla", "megatron", "attention", "adapters"]
---

The point of a shared MLA adapter is not abstraction for its own sake. The
point is to contain drift.

MLA support tends to create pressure in exactly the wrong places: layer-spec
construction, positional handling, attention-module selection, and pipeline
offset plumbing. If those conditions spread through the generic builder path,
every later attention change becomes harder to reason about. A shared adapter
is the cheaper boundary. It normalizes the MLA-only pieces and leaves the rest
of the stack boring.

## What problem this boundary solves

MegaCpp's public sample is intentionally small, but it encodes a real design
rule: keep MLA-specific compatibility in one adapter contract.

That matters because upstream Megatron surfaces in this area are not static.
Recent Megatron-LM issue reports show exactly the kind of drift a narrow seam is
meant to isolate: MLA mode has had mismatches around local layer-spec assembly
and around which attention implementation paths actually support MLA cleanly.
The right response is not to smear MLA conditionals across the codebase. The
right response is to keep one shared adapter surface that can absorb those
moving pieces. citeturn0search0turn0search5

## Why one shared adapter is safer than many tiny special cases

There are three practical wins.

- reviewability improves, because the MLA contract has a clear home
- upstream upgrades get cheaper, because compatibility edits stay localized
- public documentation gets more honest, because we can point to one visible
  boundary instead of implying MLA is just a generic attention toggle

This is the same architectural reason people isolate position-bias or
cross-entropy fusion boundaries instead of threading ad hoc switches through the
whole model stack. Once a feature changes the contract of layer construction,
the safest default is to contain it.

## Example -> article -> upstream docs

- example: [`mla_shared_adapter_sample.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mla_shared_adapter_sample.py)
- related article: [`public-safe-mla-integration-patterns-for-megatron.md`](https://megacpp.com/blog/public-safe-mla-integration-patterns-for-megatron/)
- upstream docs: Megatron-LM config and MLA bug reports around layer-spec and attention-path support citeturn0search3turn0search0turn0search5

## References

- [Shared MLA adapter sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mla_shared_adapter_sample.py)
- [MLA integration pattern sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mla_integration_pattern_sample.py)
- [Public-safe MLA integration patterns for Megatron](https://megacpp.com/blog/public-safe-mla-integration-patterns-for-megatron/)
- [Megatron-LM transformer config](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py) citeturn0search3
- [Megatron-LM MLA layer-spec bug report](https://github.com/NVIDIA/Megatron-LM/issues/1589) citeturn0search0
- [Megatron-LM FlashAttention / MLA support bug report](https://github.com/NVIDIA/Megatron-LM/issues/1698) citeturn0search5
