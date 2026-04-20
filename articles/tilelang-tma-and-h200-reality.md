---
title: "TileLang TMA and H200 reality"
description: "Why TileLang shared-memory legality and TMA lowering on Hopper-class GPUs should be treated as concrete compiler contracts rather than assumed backend magic."
date: "2026-04-19"
tags: ["tilelang", "H200", "hopper", "kernels", "tma"]
---

The useful way to talk about TileLang on H200 is not to ask whether the kernel
is mathematically correct. The useful question is whether the lowering accepts
the shared-memory layout and TMA path the kernel actually requests.

That is why MegaCpp keeps small legality-style samples. A compact reproducer is
often more valuable than one more benchmark chart when the problem lives in the
compiler contract.

## References

- [Mamba3 3D to 2D SMEM sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_sample.py)
- [TileLang TMA bulk copy SMEM sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_sample.py)
- [Upstream PR TileLang and Megatron article](https://megacpp.com/blog/upstream-pr-tilelang-and-megatron/)
