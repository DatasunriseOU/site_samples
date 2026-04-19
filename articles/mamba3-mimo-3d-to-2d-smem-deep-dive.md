---
title: "Mamba3 MIMO 3D-to-2D shared-memory deep dive"
description: "Why some Mamba3-style kernels need an explicit 3D-to-2D shared-memory legality rewrite before the backend will accept the tile layout."
date: "2026-04-19"
tags: ["mamba3", "smem", "tilelang", "kernels", "deep-dive"]
---

The deeper point of the 3D-to-2D shared-memory example is that legality is its
own contract. A kernel can be conceptually correct and still fail because the
backend accepts only a narrower shared-memory layout than the model-side code
first produced.

The near-copy example keeps that lesson visible without dragging in the whole
training stack.

The practical split is useful:

- the compact example explains the flattening rule
- the near-copy example preserves the actual shape surfaces that had to be
  rewritten: Q/K shared-memory staging and the `qk_dot_shared` tile

That makes it easier to see why this was not a math rewrite. It was a layout
rewrite done to unlock the intended lowering path.

## Example -> article -> upstream docs

- example: [`mamba3_mimo_3d_to_2d_smem_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_nearcopy.py)
- article: [`tilelang-tma-and-h200-reality.md`](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- upstream docs: TileLang TMA notes, CUDA C programming guidance for shared memory, and the surrounding MegaCpp POC runtime article

## References

- [Mamba3 MIMO 3D-to-2D SMEM near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_nearcopy.py)
- [Mamba3 MIMO 3D-to-2D SMEM compact example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_mimo_3d_to_2d_smem_sample.py)
- [TileLang TMA and H200 reality](https://megacpp.com/blog/tilelang-tma-and-h200-reality/)
- [CUDA C programming guide: shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA C programming guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
