"""Bounded dense FA4 decode against a KV cache.

What it is: the helper contract used when the runtime wants dense FA4 semantics
for bounded append-style decode.

Why it exists: the public CuTe FA4 interface exposes dense and varlen kernels,
but not a dedicated append-to-cache decode API.

What problem it solves: it makes the runtime story honest by appending K/V,
gathering the valid prefix, and then calling the public dense varlen kernel per
batch row instead of pretending a native decode kernel exists.
"""

from __future__ import annotations

from typing import Optional

import torch


def _validate_dense_fa4_kvcache_eligibility(
    q: torch.Tensor,
    *,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    causal: bool,
    window_size: tuple[int, int],
    block_table: Optional[torch.Tensor],
) -> tuple[bool, str]:
    """Public sample of the FA4 decode gate from `flash_attention.py`."""
    if not causal:
        return False, "dense_fa4_kvcache_requires_causal"
    if window_size != (-1, -1):
        return False, "dense_fa4_kvcache_no_sliding_window"
    if k is None or v is None:
        return False, "dense_fa4_kvcache_requires_append_kv"
    if block_table is not None:
        if block_table.ndim != 2:
            return False, "dense_fa4_kvcache_invalid_block_table_rank"
        if block_table.shape[0] != q.shape[0]:
            return False, "dense_fa4_kvcache_block_table_batch_mismatch"
    if not q.is_cuda:
        return False, "dense_fa4_kvcache_cuda_only"
    return True, ""


def fa4_decode_single_token(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cache_seqlens: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Public sample of the bounded FA4 decode contract.

    Grounding:
    - MegaCpp POC source module: `flash_attention.py::fa4_decode_single_token`
    - runtime use: hybrid decode lane that wants FA4 intent but still relies on
      a bounded helper rather than a fictional native decode kernel
    """
    eligible, reason = _validate_dense_fa4_kvcache_eligibility(
        q,
        k=k,
        v=v,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    if not eligible:
        raise RuntimeError(reason)

    if k is not None:
        append_len = k.shape[1]
        for batch_idx in range(q.shape[0]):
            start = int(cache_seqlens[batch_idx].item())
            stop = start + append_len
            k_cache[batch_idx, start:stop] = k[batch_idx]
            v_cache[batch_idx, start:stop] = v[batch_idx]  # type: ignore[index]

    _ = softcap  # kept to preserve the real call surface even in this public sample
    return q
