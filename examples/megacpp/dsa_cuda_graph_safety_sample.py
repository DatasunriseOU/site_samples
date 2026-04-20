"""MegaCpp POC example: make DSA index-mask updates CUDA-graph-safe.

What this solves in simple words:
- CUDA Graph capture cannot tolerate hidden CPU sync points in the hot path.
- Branching on GPU reductions such as ``torch.any(... )`` or validation checks
  that force a Python bool can break capture even when the math is correct.
- The safe pattern is branchless scatter plus a small fixup instead of host-side
  conditionals during capture.

This sample is adapted from the MegaCpp upstream reproducer pack with only
public-safe cleanup applied.
"""

from __future__ import annotations

import torch


class DsaScatterGraphSafe(torch.nn.Module):
    """Branchless index-mask update suitable for CUDA Graph capture."""

    def __init__(self, s_kv: int) -> None:
        super().__init__()
        self.s_kv = s_kv

    def forward(
        self,
        index_mask: torch.Tensor,
        idx_chunk: torch.Tensor,
        s0: int,
        s1: int,
    ) -> torch.Tensor:
        sentinel = idx_chunk < 0
        safe_chunk = idx_chunk.clamp(min=0)
        index_mask[:, s0:s1].scatter_(-1, safe_chunk, 0.0)

        has_sentinel = sentinel.any(dim=-1)
        has_real_zero = ((idx_chunk == 0) & ~sentinel).any(dim=-1)
        fixup = has_sentinel & ~has_real_zero
        index_mask[:, s0:s1, 0].masked_fill_(fixup, float("-inf"))
        return index_mask


def eager_reference(
    index_mask: torch.Tensor,
    idx_chunk: torch.Tensor,
    s0: int,
    s1: int,
) -> torch.Tensor:
    """Reference branchy path for parity checking outside graph capture."""

    out = index_mask.clone()
    valid = idx_chunk >= 0
    if valid.any().item():
        b_idx, q_rel_idx, t_idx = torch.where(valid)
        q_idx = q_rel_idx + s0
        k_idx = idx_chunk[b_idx, q_rel_idx, t_idx]
        out[b_idx, q_idx, k_idx] = 0.0
    return out
