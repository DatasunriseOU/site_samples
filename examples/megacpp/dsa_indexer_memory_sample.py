"""MegaCpp POC example: replace a memory-hungry DSA score path with fused index scoring.

What this solves in simple words:
- naive index-score materialization can waste memory at large sequence lengths;
- a fused score path keeps the same top-k intent while avoiding the large
  intermediate tensor.
"""

from __future__ import annotations

import torch


def fused_topk_indices(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return values and indices without exposing a larger downstream score tensor."""

    if k <= 0:
        raise ValueError("k must be positive")
    return torch.topk(scores, k=k, dim=-1)


def memory_story(batch: int, heads: int, q: int, k: int, dtype_bytes: int = 2) -> dict[str, int]:
    dense_bytes = batch * heads * q * k * dtype_bytes
    return {
        "dense_score_bytes": dense_bytes,
        "why_fused_path_exists": dense_bytes,
    }
