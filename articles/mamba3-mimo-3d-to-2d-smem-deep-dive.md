---
title: "Mamba3 MIMO 3D-to-2D shared-memory deep dive"
description: "Why some Mamba3-style kernels need an explicit 3D-to-2D shared-memory legality rewrite before the backend will accept the tile layout."
date: "2026-04-19"
tags: ["mamba3", "smem", "tilelang", "kernels", "deep-dive"]
---

The deeper point of the 3D-to-2D shared-memory example is that layout legality
is its own contract. A kernel can be conceptually correct and still fail
because backend lowering may accept a narrower shared-memory layout than the
original kernel-side tensor view.

The near-copy example keeps that lesson visible without dragging in the whole
training stack.

The practical split is useful:

- the compact example explains the flattening rule
- the near-copy example preserves the actual shared-memory and indexing
  surfaces that had to be rewritten: Q/K shared-memory staging and the
  `qk_dot_shared` tile

That makes it easier to see why this was not a math rewrite. It was a layout
rewrite done to preserve the same indexing semantics while making the
shared-memory view easier for the lowering path to accept.

## Example -> article -> upstream docs

- example: [`mamba3_mimo_3d_to_2d_smem_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_nearcopy.py)
- article: this deep-dive route, [`mamba3-mimo-3d-to-2d-smem-deep-dive`](https://megacpp.com/blog/mamba3-mimo-3d-to-2d-smem-deep-dive/), plus the broader context article [`tilelang-tma-and-h200-reality`](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- upstream docs: CUDA shared-memory and TMA guidance, CuTe TMA tensor docs, and the surrounding MegaCpp POC runtime article

## References

- [Mamba3 MIMO 3D-to-2D SMEM near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_nearcopy.py)
- [Mamba3 MIMO 3D-to-2D SMEM compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_sample.py)
- [TileLang TMA and H200 reality](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- [CUDA C programming guide: shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA C programming guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
- [CUTLASS CuTe TMA tensors](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html)
- [Hopper PTX / tensor-memory description](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/parallel-thread-execution/index.html)
