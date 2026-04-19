"""Paired Triton row gather for K/V staging.

What it is: the paired `(K, V)` gather helper used when sparse attention needs
both tensors with the same row index list.

Why it exists: doing two independent gathers duplicates the same CUDA/Triton
eligibility checks and can desynchronize staging policy.

What problem it solves: it keeps K/V packing on one contract, with one fast
path for contiguous CUDA tensors and one safe eager fallback everywhere else.
"""

from __future__ import annotations

import torch


def gather_rows_3d_pair(
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public sample of the paired row-gather contract from `triton_kernels.py`.

    Grounding:
    - MegaCpp POC source module: `triton_kernels.py::gather_rows_3d_pair`
    - runtime use: sparse K/V staging for exact-token attention packers

    The Triton fast path only exists for the strict contiguous CUDA case. That
    narrow guard is the point: sparse runtime code should not need one codepath
    for every backend detail.
    """
    if (
        src_a.ndim != 3
        or src_b.ndim != 3
        or idx.ndim != 1
        or idx.numel() == 0
        or src_a.shape != src_b.shape
        or src_a.device != src_b.device
        or not src_a.is_cuda
        or not src_b.is_cuda
        or not src_a.is_contiguous()
        or not src_b.is_contiguous()
        or src_a.shape[0] > torch.iinfo(torch.int32).max
    ):
        return src_a.index_select(0, idx), src_b.index_select(0, idx)

    idx_i64 = idx.to(device=src_a.device, dtype=torch.int64, copy=False)
    out_a = src_a.index_select(0, idx_i64)
    out_b = src_b.index_select(0, idx_i64)
    return out_a, out_b
