"""Near-copy MegaCpp POC example: DSA CUDA graph safety reproducer.

This keeps the real contract close to the original reproducer:
- an unpatched path branches on GPU reductions and uses a validation check that
  forces a Python bool;
- a patched path uses branchless scatter plus sentinel fixup.

Only public-safe cleanup was applied.
"""

from __future__ import annotations

import torch


class DsaScatterUnpatched(torch.nn.Module):
    def __init__(self, s_kv: int) -> None:
        super().__init__()
        self.s_kv = s_kv

    def forward(
        self,
        index_mask: torch.Tensor,
        idx_chunk: torch.Tensor,
        finite_ref: torch.Tensor,
        finite_got: torch.Tensor,
        s0: int,
        s1: int,
    ) -> torch.Tensor:
        if not torch.equal(finite_got, finite_ref):
            raise RuntimeError("finite mask mismatch")
        if torch.any(idx_chunk < 0):
            valid_topk = idx_chunk >= 0
            if valid_topk.any():
                b_idx, q_rel_idx, t_idx = torch.where(valid_topk)
                q_idx = q_rel_idx + s0
                k_idx = idx_chunk[b_idx, q_rel_idx, t_idx]
                index_mask[b_idx, q_idx, k_idx] = 0.0
        else:
            index_mask[:, s0:s1].scatter_(-1, idx_chunk, 0.0)
        return index_mask


class DsaScatterPatched(torch.nn.Module):
    def __init__(self, s_kv: int) -> None:
        super().__init__()
        self.s_kv = s_kv

    def forward(
        self,
        index_mask: torch.Tensor,
        idx_chunk: torch.Tensor,
        finite_ref: torch.Tensor,
        finite_got: torch.Tensor,
        s0: int,
        s1: int,
    ) -> torch.Tensor:
        sentinel = idx_chunk < 0
        safe_chunk = idx_chunk.clamp(min=0)
        index_mask[:, s0:s1].scatter_(-1, safe_chunk, 0.0)
        has_sent = sentinel.any(dim=-1)
        has_real0 = ((idx_chunk == 0) & ~sentinel).any(dim=-1)
        fixup = has_sent & ~has_real0
        index_mask[:, s0:s1, 0].masked_fill_(fixup, float("-inf"))
        return index_mask


def eager_reference(index_mask: torch.Tensor, idx_chunk: torch.Tensor, s0: int, s1: int) -> torch.Tensor:
    out = index_mask.clone()
    valid = idx_chunk >= 0
    if valid.any().item():
        b_idx, q_rel_idx, t_idx = torch.where(valid)
        q_idx = q_rel_idx + s0
        k_idx = idx_chunk[b_idx, q_rel_idx, t_idx]
        out[b_idx, q_idx, k_idx] = 0.0
    return out
