---
title: "Sparse MLA FP8 dispatch"
description: "Why SparseMLA needs an FP8-aware dispatch contract when Transformer Engine wrappers hide FP8 storage behind a bf16-looking logical surface."
date: "2026-04-19"
tags: ["sparse-mla", "fp8", "transformer-engine", "dispatch"]
---

The failure here is not just "FP8 is hard." The real problem is that generic
dispatch logic can mistake a wrapper type for an ordinary dense tensor.

In the SparseMLA path, that matters because the wrapper can report a logical
bf16-looking surface while the real storage is FP8 and the raw pointer surface
is not what a naive kernel dispatch expects. That creates two bad outcomes:

- hard failure, if the kernel reaches a NULL-facing pointer surface
- silent downgrade, if the wrapper gets dequantized implicitly and the bf16 path
  runs instead of the requested FP8 path

The public sample keeps all three lanes visible: raw dispatch, dequantize
fallback, and explicit FP8-aware dispatch.

## Why the explicit FP8 path matters

The dequantize fallback is a real fix for correctness, but it is not the same
thing as an FP8-aware runtime contract. A fallback path pays extra movement and
can silently erase the reason FP8 was enabled in the first place. The better
public design is to keep dispatch honest: if the input is a quantized FP8
wrapper, route it to the FP8-capable kernel surface explicitly.

That is the same basic engineering rule as the CUDA-graph and TileLang examples
already in this pack. The bug is not abstract. The bug is that one runtime
surface lies about the contract another runtime surface actually needs.

## Example -> article -> upstream docs

- example: [`sparse_mla_fp8_dispatch_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/sparse_mla_fp8_dispatch_nearcopy.py)
- article: [`sparse-mla-fp8-dispatch`](https://megacpp.com/blog/sparse-mla-fp8-dispatch/)
- upstream docs: Transformer Engine float8 tensors, NVIDIA Transformer Engine overview, and Megatron-LM repository context

## References

- [Sparse MLA FP8 dispatch near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/sparse_mla_fp8_dispatch_nearcopy.py)
- [Transformer Engine repository](https://github.com/NVIDIA/TransformerEngine)
- [Transformer Engine user guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
