---
title: "Sparse MLA dimension generalization"
description: "Why SparseMLA kernels that hardcode DeepSeek-sized dimensions fail to scale down cleanly to NAM56R-style shapes, and what a generalized contract changes."
date: "2026-04-19"
tags: ["sparse-mla", "dimensions", "kernels", "nam56r"]
---

This class of bug looks like a numerical issue from the outside, but the first
real failure is simpler: one kernel family assumes a fixed dimension contract.

If a SparseMLA path hardcodes one set of dimensions for QK and V channels, it
can pass on one model family and fail on another even when the algorithm is the
same. That is why the public example compares a DeepSeek-shaped hardcoded lane
with a generalized lane that accepts smaller NAM56R-style dimensions.

## What the generalized path is actually fixing

The generalized path is not inventing a new kernel idea. It is removing fixed
dimension assumptions from the contract surface and threading the real values
through forward and backward plumbing.

That distinction matters. Once `d_total` and `d_v` become parameters instead of
magic constants, the same sparse MLA idea can survive outside the original
shape family it was first authored around.

## Example -> article -> upstream docs

- example: [`sparse_mla_dimension_generalization_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/sparse_mla_dimension_generalization_nearcopy.py)
- article: [`sparse-mla-dimension-generalization`](https://megacpp.com/blog/sparse-mla-dimension-generalization/)
- upstream docs: Megatron-LM repository context plus sparse-attention and MLA implementation references

## References

- [Sparse MLA dimension generalization near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/sparse_mla_dimension_generalization_nearcopy.py)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [FlashMLA repository](https://github.com/deepseek-ai/FlashMLA)
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)
