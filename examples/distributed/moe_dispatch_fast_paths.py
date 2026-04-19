"""MoE dispatch fast-path sample.

This example summarizes the MegaCpp POC dispatch fast paths that reduce routing
overhead. The problem they solve is that token permutation and all-to-all can
eat a large part of step time if the dispatch path stays purely generic.
"""

from __future__ import annotations

import os


def truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def dispatch_fast_paths() -> dict[str, bool]:
    """Expose the MegaCpp POC-style optional fast paths for MoE dispatch."""
    return {
        "deep_ep": truthy_env("MEGACPP_MOE_DEEP_EP"),
        "te_permute": truthy_env("MEGACPP_MOE_TE_PERMUTE"),
        "megatron_dispatch": truthy_env("MEGACPP_USE_MEGATRON_DISPATCH"),
        "async_alltoall": truthy_env("MEGACPP_MOE_ASYNC_ALLTOALL"),
    }


def explain_dispatch_problem() -> tuple[str, ...]:
    return (
        "routing needs token reordering before experts can run",
        "all-to-all exchange can dominate the hot path at larger expert counts",
        "specialized fused permute and overlap paths cut that dispatch overhead",
    )
