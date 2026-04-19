"""Near-copy MegaCpp POC example: TileLang TMA bulk-copy layout reproducer.

This keeps the contract close to the real repro pack: one path describes the
natural 3D shared-memory destination, and the other path shows the flattened 2D
destination that better matches the lowering path. The intended data motion is
the same. The compiler surface is different.
"""

from __future__ import annotations


def bulk_copy_3d_contract(d0: int, d1: int, d2: int) -> dict[str, tuple[int, ...]]:
    if min(d0, d1, d2) <= 0:
        raise ValueError("all shape dims must be positive")
    return {
        "global_shape": (d0, d1, d2),
        "shared_shape": (d0, d1, d2),
        "rank": (d0, d1, d2),
    }


def bulk_copy_2d_contract(d0: int, d1: int, d2: int) -> dict[str, tuple[int, ...]]:
    if min(d0, d1, d2) <= 0:
        raise ValueError("all shape dims must be positive")
    return {
        "global_shape": (d0 * d1, d2),
        "shared_shape": (d0 * d1, d2),
        "rank": (d0 * d1, d2),
    }


def lowered_width(d0: int, d1: int, d2: int) -> int:
    return bulk_copy_2d_contract(d0, d1, d2)["shared_shape"][1]


def tma_path_is_supported(d0: int, d1: int, d2: int, max_width: int = 256) -> bool:
    return lowered_width(d0, d1, d2) <= max_width


def compare_layouts(d0: int, d1: int, d2: int) -> dict[str, object]:
    return {
        "three_dimensional": bulk_copy_3d_contract(d0, d1, d2),
        "flattened_two_dimensional": bulk_copy_2d_contract(d0, d1, d2),
        "tma_supported": tma_path_is_supported(d0, d1, d2),
    }
