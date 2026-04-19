---
title: "TileLang TMA bulk-copy 3D shared-memory deep dive"
description: "A deeper reproducer-driven look at why TileLang TMA bulk-copy paths can fail on shared-memory layout legality before the math is even the problem."
date: "2026-04-19"
tags: ["tilelang", "tma", "hopper", "smem", "deep-dive"]
---

This class of bug matters because it looks like a kernel problem but is often a
lowering problem. The near-copy example keeps the shape of that failure close
to the original repro pack: the payload is fine, the shared-memory contract is not.

That distinction matters on Hopper-class hardware. People often assume TMA
trouble means the kernel math is wrong or the copy is too large. In practice,
the first failure can happen much earlier: the compiler cannot prove the layout
contract it needs for the TMA path, so it rejects the form or falls back.

## Example -> article -> upstream docs

- example: [`tilelang_tma_bulk_copy_smem_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_nearcopy.py)
- article: [`tilelang-tma-and-h200-reality.md`](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- upstream docs: CUDA TMA guidance, TileLang-facing lowering context, and the surrounding MegaCpp POC article

## References

- [TileLang TMA bulk-copy SMEM near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_nearcopy.py)
- [TileLang TMA bulk-copy SMEM compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_sample.py)
- [TileLang TMA and H200 reality](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- [CUDA C programming guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
- [NVIDIA Hopper architecture overview](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
