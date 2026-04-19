---
title: "TileLang TMA bulk-copy 3D shared-memory deep dive"
description: "A deeper reproducer-driven look at why TileLang TMA bulk-copy paths can fail on shared-memory layout legality before the math is even the problem."
date: "2026-04-19"
tags: ["tilelang", "tma", "hopper", "smem", "deep-dive"]
---

This class of bug matters because it looks like a kernel problem but is often a
lowering problem. The near-copy example keeps the shape of that failure close
to the original repro pack: the intended data movement is fine, but the
shared-memory/TMA layout contract is not.

That distinction matters on Hopper-class hardware. It is easy to misdiagnose
this as a kernel-math problem rather than a layout/lowering problem. In
practice, the first failure can happen much earlier: the lowering cannot map
the requested form onto the expected TMA/shared-memory contract, so the form is
rejected or de-optimized before the intended fast path is even in play.

## Example -> article -> upstream docs

- example: [`tilelang_tma_bulk_copy_smem_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_nearcopy.py)
- article: this deep-dive route, [`tilelang-tma-bulk-copy-3d-smem-deep-dive`](https://megacpp.com/blog/tilelang-tma-bulk-copy-3d-smem-deep-dive/), plus the broader context article [`tilelang-tma-and-h200-reality`](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- upstream docs: CUDA TMA guidance, TileLang-facing lowering context, and the surrounding MegaCpp POC article

## References

- [TileLang TMA bulk-copy SMEM near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_nearcopy.py)
- [TileLang TMA bulk-copy SMEM compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/tilelang_tma_bulk_copy_smem_sample.py)
- [TileLang TMA and H200 reality](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- [CUDA C programming guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
- [NVIDIA Hopper architecture overview](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Hopper tuning guide](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html)
- [TileLang InjectFenceProxy internals](https://www.tilelang.com/compiler_internals/inject_fence_proxy.html)
