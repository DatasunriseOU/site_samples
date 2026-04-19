"""Null-slot routing capacity sample.

This example shows how the MegaCpp POC models optional null routing slots.
The problem it solves is wasted expert compute on tokens that should rely only
on the shared expert path.
"""

from __future__ import annotations

import math


def null_slot_plan(n_routed: int, top_k: int, null_rho: float) -> dict[str, int | float]:
    """Compute the expanded routing pool used by the null-slot token-choice path."""
    if n_routed <= 0:
        raise ValueError("n_routed must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not (0.0 < null_rho <= 1.0):
        raise ValueError("null_rho must be in (0, 1]")

    if null_rho < 1.0:
        n_null_copies = max(1, int(n_routed * (1.0 - null_rho) / null_rho))
        k_max = math.ceil(top_k / null_rho)
    else:
        n_null_copies = 0
        k_max = top_k

    expanded_pool_size = n_routed + n_null_copies
    return {
        "n_routed": n_routed,
        "top_k": top_k,
        "null_rho": null_rho,
        "n_null_copies": n_null_copies,
        "expanded_pool_size": expanded_pool_size,
        "k_max": min(k_max, expanded_pool_size),
    }
