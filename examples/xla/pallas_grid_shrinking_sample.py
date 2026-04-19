"""Pallas grid shrinking sample.

What it is: a donor-based excerpt of the mask compaction step used before the
TPU Pallas softcapped attention custom call.
Why it exists: sparse masks often contain many empty KV columns, and iterating
over all of them wastes TPU work and scalar-memory budget.
What problem it solves: it shrinks the iteration grid to the non-zero block
columns only, then passes the compact index tables into the kernel as scalar
prefetch inputs.
"""

from __future__ import annotations

from typing import cast

import numpy as np

type SplashInfoResult = tuple[np.ndarray, np.ndarray, int, np.ndarray]
type SplashInfoTResult = tuple[np.ndarray, np.ndarray, int]

_splash_info_cache: dict[tuple[object, ...], object] = {}


def _build_splash_info(mask_obj, q_seq_len, kv_seq_len, block_q, block_k) -> SplashInfoResult:
    """Build grid-shrunk mask info for forward and dQ passes."""
    cache_key = ("fwd", mask_obj, q_seq_len, kv_seq_len, block_q, block_k)
    if cache_key in _splash_info_cache:
        return cast(SplashInfoResult, _splash_info_cache[cache_key])

    nq = q_seq_len // block_q
    full_bm = mask_obj.build_block_mask(q_seq_len, kv_seq_len, block_q, block_k)

    full_bm, partial_mask_blocks = _build_partial_mask_blocks(
        mask_obj, full_bm, q_seq_len, kv_seq_len, block_q, block_k
    )

    max_nnz = 0
    row_nnz = []
    for qi in range(nq):
        nnz = np.nonzero(full_bm[qi])[0]
        row_nnz.append(nnz)
        max_nnz = max(max_nnz, len(nnz))

    grid_width = max(max_nnz, 1)
    block_mask = np.zeros((nq, grid_width), dtype=np.int32)
    data_next = np.zeros((nq, grid_width), dtype=np.int32)

    for qi in range(nq):
        for j, kv_idx in enumerate(row_nnz[qi]):
            block_mask[qi, j] = full_bm[qi, kv_idx]
            data_next[qi, j] = kv_idx
    result = (block_mask, data_next, grid_width, partial_mask_blocks)
    _splash_info_cache[cache_key] = result
    return result


def _build_splash_info_t(mask_obj, q_seq_len, kv_seq_len, block_q, block_k) -> SplashInfoTResult:
    """Build transposed grid-shrunk mask info for dKV backward."""
    cache_key = ("bwd", mask_obj, q_seq_len, kv_seq_len, block_q, block_k)
    if cache_key in _splash_info_cache:
        return cast(SplashInfoTResult, _splash_info_cache[cache_key])

    nk = kv_seq_len // block_k
    full_bm = mask_obj.build_block_mask(q_seq_len, kv_seq_len, block_q, block_k)
    full_bm, _ = _build_partial_mask_blocks(
        mask_obj, full_bm, q_seq_len, kv_seq_len, block_q, block_k
    )
    full_bm_t = full_bm.T.copy()

    max_nnz = 0
    row_nnz = []
    for ki in range(nk):
        nnz = np.nonzero(full_bm_t[ki])[0]
        row_nnz.append(nnz)
        max_nnz = max(max_nnz, len(nnz))

    grid_width_t = max(max_nnz, 1)
    block_mask_t = np.zeros((nk, grid_width_t), dtype=np.int32)
    data_next_t = np.zeros((nk, grid_width_t), dtype=np.int32)

    for ki in range(nk):
        for j, q_idx in enumerate(row_nnz[ki]):
            block_mask_t[ki, j] = full_bm_t[ki, q_idx]
            data_next_t[ki, j] = q_idx
    result = (block_mask_t, data_next_t, grid_width_t)
    _splash_info_cache[cache_key] = result
    return result


def _build_partial_mask_blocks(mask_obj, full_bm, q_seq_len, kv_seq_len, block_q, block_k):
    """Placeholder dependency for publication pack wiring.

    The donor implementation precomputes dense partial masks here. Public docs
    link this file for the grid-shrinking contract, not for the full mask
    materialization routine.
    """
    partial_mask_blocks = np.zeros((0, block_q, block_k), dtype=np.int32)
    return full_bm, partial_mask_blocks

