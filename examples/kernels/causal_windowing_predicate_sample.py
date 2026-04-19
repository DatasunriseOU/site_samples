"""Causal windowing predicate sample.

What it is: a public-safe receipt for the causal bound used inside sparse
Pallas fused scoring.

Why it exists: the scoring kernel tiles over compressed blocks in order, so it
can stop once future tiles can no longer contribute to a causal query chunk.

What problem it solves: it shows how the sparse TPU path gets a free speedup
without changing recall, simply by refusing to score future-only tiles.
"""

from __future__ import annotations


def can_score_tile(*, tile_start: int, block_stride: int, q_chunk_end: int) -> bool:
    """Return True when a compressed tile is still causal for this query chunk."""

    return int(tile_start) * int(block_stride) <= int(q_chunk_end)


def summarize_causal_windowing(*, scoring_tile_size: int, block_stride: int, q_chunk_start: int, q_chunk_size: int) -> dict[str, object]:
    q_chunk_end = int(q_chunk_start) + int(q_chunk_size) - 1
    active_tiles: list[int] = []
    skipped_tiles: list[int] = []
    for tile_start in range(0, scoring_tile_size * 4, scoring_tile_size):
        if can_score_tile(tile_start=tile_start, block_stride=block_stride, q_chunk_end=q_chunk_end):
            active_tiles.append(tile_start)
        else:
            skipped_tiles.append(tile_start)
    return {
        "q_chunk_end": q_chunk_end,
        "block_stride": int(block_stride),
        "active_tiles": active_tiles,
        "skipped_future_tiles": skipped_tiles,
        "rule": "stop scoring when tile_start * block_stride exceeds q_chunk_end",
    }
