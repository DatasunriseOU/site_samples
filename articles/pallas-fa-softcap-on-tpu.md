---
title: "Pallas FlashAttention with logit softcap on TPU v6e"
description: "Why softcap attention on TPU needs a dedicated kernel surface: fuse the nonlinearity, keep masking contract-friendly, and avoid turning a stability trick into a second full pass over the score matrix."
date: "2026-04-18"
tags: ["pallas", "tpu", "v6e", "flash-attention", "softcap", "doc-masking"]
---

Softcap attention is a good example of when TPU kernel work becomes justified. The point is not to replace every stock kernel. The point is to avoid paying for an extra pass over the score matrix when a fused path can keep the stability trick inside the main attention loop.

## Why softcap needs special treatment

Softcap changes attention logits before softmax. On paper the math is simple. In practice, an unfused implementation can turn one useful nonlinearity into more memory traffic and another expensive walk over the score tiles.

That is why the public TPU claim should stay narrow: use a custom Pallas path when softcap, masking, and local-window behavior together justify it.

## What the kernel should keep explicit

A good TPU softcap kernel keeps four things explicit:

- softcap fused into the main score path
- local-window behavior expressed without dynamic per-step retracing
- masking carried as stable metadata rather than ad hoc dense materialization
- document-boundary handling kept compatible with packed-sequence segment identifiers

The point is not one exact implementation detail. The point is a stable compiled contract.

## Why masking contract matters here too

Softcap alone is not the whole kernel story. The moment local windows, segment boundaries, or sparse grid decisions enter the picture, the kernel stops being a pure numerical tweak and becomes part of the model's masking contract. That is where a custom TPU path becomes easier to justify.

## What should stay out

A TPU softcap path becomes harder to trust if it also tries to absorb every adjacent idea. In practice, it is safer to avoid:

- dynamic per-step window changes
- optional fallback trees with radically different semantics
- unrelated fused epilogues that do not clearly pay for themselves

## The useful public summary

The public statement is simple: MegaCpp uses a selective Pallas softcap attention path on TPU where fused softcap and stable masking semantics buy a real runtime win. It does not present that path as the default answer to every TPU attention problem.

## References

- https://docs.jax.dev/en/latest/pallas/tpu/
- https://docs.jax.dev/en/latest/pallas/tpu/details.html
- https://arxiv.org/abs/2307.08691
- https://blog.google/technology/developers/google-gemma-2/
- https://github.com/DatasunriseOU/site_samples/blob/main/docs/pallas-kernel-selection.md
