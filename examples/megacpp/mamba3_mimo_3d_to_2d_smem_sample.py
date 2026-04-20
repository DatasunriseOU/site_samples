"""MegaCpp POC example: flatten a 3D shared-memory view into a legal 2D tile.

What this solves in simple words:
- some kernel DSLs and lowerings accept only a narrower shared-memory layout
  than the model-side logic would naturally produce;
- a small explicit reshape can preserve the same logical payload while making
  the tile layout legal for the backend.
"""

from __future__ import annotations


def flatten_3d_tile_to_2d(depth: int, rows: int, cols: int) -> tuple[int, int]:
    """Convert a logical ``(depth, rows, cols)`` tile into a 2D shared-memory tile."""

    if depth <= 0 or rows <= 0 or cols <= 0:
        raise ValueError("tile dimensions must be positive")
    return depth * rows, cols


def tile_is_layout_compatible(depth: int, rows: int, cols: int, max_cols: int = 256) -> bool:
    """Small legality guard for examples and tests."""

    flat_rows, flat_cols = flatten_3d_tile_to_2d(depth, rows, cols)
    return flat_rows > 0 and 0 < flat_cols <= max_cols
