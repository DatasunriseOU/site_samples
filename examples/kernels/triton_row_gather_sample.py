"""Triton row-gather staging for sparse attention.

What it is: the narrow CUDA fast path used to stage selected K/V rows from a
contiguous `(N, H, D)` cache.

Why it exists: exact-token sparse attention needs to gather many irregular rows
without forcing every caller to know whether Triton is available.

What problem it solves: it keeps one stable contract while using a Triton
kernel only when the input is CUDA, contiguous, and small enough to stay on the
proven path; otherwise it falls back to `index_select`.
"""

from __future__ import annotations

import torch


def gather_rows_3d(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Public sample of the sparse attention row-staging contract.

    Grounding:
    - MegaCpp POC source: `triton_kernels.py::gather_rows_3d`
    - runtime caller source: `sparse_attention.py::_gather_sparse_rows_3d`

    The important change is not “use Triton everywhere”. The runtime only uses
    the CUDA helper when the tensor layout matches the kernel assumptions and
    when the row count is still inside the bounded staging lane. Outside those
    conditions the implementation deliberately keeps the eager fallback.
    """
    if (
        src.ndim != 3
        or idx.ndim != 1
        or idx.numel() == 0
        or not src.is_cuda
        or not src.is_contiguous()
        or src.shape[0] > torch.iinfo(torch.int32).max
    ):
        return src.index_select(0, idx)

    rows = idx.numel()
    inner_dim = src.shape[1] * src.shape[2]
    src_flat = src.reshape(src.shape[0], inner_dim)
    idx_i32 = idx.to(device=src.device, dtype=torch.int32, copy=False)
    out = torch.empty((rows, src.shape[1], src.shape[2]), device=src.device, dtype=src.dtype)
    out_flat = out.reshape(rows, inner_dim)

    # The MegaCpp POC launches a Triton kernel here after choosing BLOCK_SIZE and
    # num_warps from `inner_dim`. This sample keeps the exact staging contract
    # but leaves the actual Triton launch out so the file remains import-safe.
    out_flat.copy_(src_flat.index_select(0, idx_i32.to(dtype=torch.int64)))
    return out


def gather_rows_3d_pair(
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused two-tensor variant used for paired K/V staging.

    Grounding:
    - MegaCpp POC source: `triton_kernels.py::gather_rows_3d_pair`

    The change relative to two separate gathers is that the MegaCpp POC kernel stages
    both tensors behind one row-layout check and one kernel launch. That keeps
    K/V staging aligned and avoids duplicated eligibility logic in the sparse
    attention packer.
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
    return src_a.index_select(0, idx_i64), src_b.index_select(0, idx_i64)


def sparse_attention_gather_policy(idx: torch.Tensor) -> str:
    """Runtime policy lifted from the MegaCpp POC sparse-attention wrapper.

    `sparse_attention.py` keeps Triton only for bounded chunked staging. Once a
    request grows past one million rows, the runtime intentionally falls back to
    `index_select` until the giant-row lane has a real receipt.
    """
    if idx.numel() > 1_000_000:
        return "fallback:index_select"
    return "fast_path:triton_row_gather"
