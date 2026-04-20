"""MegaCpp POC example: isolate a TileLang TMA bulk-copy shared-memory layout issue.

What this solves in simple words:
- TMA or bulk-copy paths can fail not because the math is wrong, but because the
  intermediate shared-memory layout is shaped in a way the lowering cannot accept;
- shrinking the problem to a small layout contract makes the regression testable.
"""

from __future__ import annotations


def tma_bulk_copy_layout(shape: tuple[int, int, int]) -> tuple[int, int]:
    """Return the lowered 2D layout a bulk-copy path expects."""

    d0, d1, d2 = shape
    if min(shape) <= 0:
        raise ValueError("shape dims must be positive")
    return d0 * d1, d2


def requires_layout_fix(shape: tuple[int, int, int], max_width: int = 256) -> bool:
    _, width = tma_bulk_copy_layout(shape)
    return width > max_width
