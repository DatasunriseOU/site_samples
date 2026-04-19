"""Near-copy MegaCpp POC example: Mamba3 MIMO 3D to 2D shared-memory refactor.

This is intentionally closer to the real reproducer than the compact example.
The issue is not the math. The issue is that a natural 3D shared-memory view can
block the TMA path, while the same payload flattened into a 2D layout keeps the
same algebra but fits the lowering contract better.
"""

from __future__ import annotations


B = 1
S = 8
R = 4
G = 1
N = 16
CHUNK_SIZE = 4
NCHUNKS = S // CHUNK_SIZE
FUSED_CHUNK_SIZE = CHUNK_SIZE * R


def flatten_qk_tile(depth: int, rows: int, cols: int) -> tuple[int, int]:
    if min(depth, rows, cols) <= 0:
        raise ValueError("all tile dimensions must be positive")
    return depth * rows, cols


def flatten_qk_dot_tile(chunk_size: int, rank: int) -> tuple[int, int]:
    if chunk_size <= 0 or rank <= 0:
        raise ValueError("chunk_size and rank must be positive")
    return chunk_size, rank * rank


def build_3d_contract() -> dict[str, tuple[int, ...]]:
    return {
        "q_shape": (B, S, R, G, N),
        "k_shape": (B, S, R, G, N),
        "q_shared": (CHUNK_SIZE, R, N),
        "k_shared": (CHUNK_SIZE, R, N),
        "qk_dot_shared": (CHUNK_SIZE, R, R),
        "out_shape": (B, NCHUNKS, CHUNK_SIZE, R, R),
    }


def build_2d_contract() -> dict[str, tuple[int, ...]]:
    q_rows, q_cols = flatten_qk_tile(CHUNK_SIZE, R, N)
    dot_rows, dot_cols = flatten_qk_dot_tile(CHUNK_SIZE, R)
    return {
        "q_shape": (B, S * R, G, N),
        "k_shape": (B, S * R, G, N),
        "q_shared": (q_rows, q_cols),
        "k_shared": (q_rows, q_cols),
        "qk_dot_shared": (dot_rows, dot_cols),
        "out_shape": (B, NCHUNKS, CHUNK_SIZE, R, R),
    }


def remap_q_index(chunk_id: int, chunk_row: int, rank_row: int) -> int:
    return chunk_id * FUSED_CHUNK_SIZE + chunk_row * R + rank_row


def remap_qk_dot_offset(chunk_row: int, rank_i: int, rank_j: int) -> tuple[int, int]:
    return chunk_row, rank_i * R + rank_j


def explain_legality(max_rank2_width: int = 256) -> dict[str, object]:
    q_rows, q_cols = flatten_qk_tile(CHUNK_SIZE, R, N)
    dot_rows, dot_cols = flatten_qk_dot_tile(CHUNK_SIZE, R)
    return {
        "3d_q_shared": build_3d_contract()["q_shared"],
        "2d_q_shared": (q_rows, q_cols),
        "3d_qk_dot_shared": build_3d_contract()["qk_dot_shared"],
        "2d_qk_dot_shared": (dot_rows, dot_cols),
        "tma_compatible_width": q_cols <= max_rank2_width and dot_cols <= max_rank2_width,
    }
