---
title: "Megatron FLCE on Hopper"
description: "Why Hopper-ready fused linear cross entropy is an output-layer contract as much as a kernel choice, and why shape-compatible alternatives are not enough."
date: "2026-04-19"
tags: ["megatron", "flce", "hopper", "cross-entropy"]
---

It is tempting to describe fused linear cross entropy as a kernel detail. That
is too shallow.

On the Hopper path, FLCE is also an output-layer contract. The model path has
to present the fused loss surface the runtime expects. A plain column-parallel
layer may be shape-compatible and still fail to preserve the intended fused
loss path.

That is why this public sample keeps the comparison narrow: one lane exposes a
plain output layer, the other exposes a fused linear-plus-cross-entropy path
that is actually aligned with the Hopper runtime contract.

## Why this matters beyond one kernel

The real engineering lesson is that parity checks have to happen at the loss
boundary, not just the tensor-shape boundary. Once a stack uses fused output
and loss handling, a "close enough" output module is not close enough.

That is the same design pressure visible in the Mamba CE parity work. The bug
surface is small, but the consequence is broad because the output path sits on
every training step.

## Example -> article -> upstream docs

- example: [`megatron_flce_hopper_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/megatron_flce_hopper_nearcopy.py)
- article: [`megatron-flce-on-hopper`](https://megacpp.com/blog/megatron-flce-on-hopper/)
- upstream docs: Megatron Core fused cross entropy docs, language-module docs, and Megatron-LM repository context

## References

- [Megatron Hopper FLCE near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/megatron_flce_hopper_nearcopy.py)
- [Megatron Core fused cross entropy docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_cross_entropy.html)
- [Megatron Core language module docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.models.common.language_module.language_module.html)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
