"""Mixture-of-Depths routing sample.

This example shows how token routing can skip expensive layers for tokens that
do not need them. The problem it solves is wasted compute on easy tokens while
still keeping the full layer available for hard ones.
"""

from __future__ import annotations

import torch


def mod_capacity(seq_len: int, capacity_factor: float) -> int:
    """Mirror the MegaCpp POC rule for how many tokens a routed layer may keep."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    capacity = int(seq_len * capacity_factor)
    return max(capacity, 1)


def mod_topk_indices(router_logits: torch.Tensor, capacity: int) -> torch.Tensor:
    """Pick the highest-scoring tokens that stay on the expensive path."""
    if router_logits.dim() != 2:
        raise ValueError("router_logits must be [batch, seq]")
    _, indices = torch.topk(router_logits, k=min(capacity, router_logits.shape[1]), dim=1, sorted=False)
    return torch.sort(indices, dim=1).values


def routed_token_mask(router_logits: torch.Tensor, capacity_factor: float) -> torch.Tensor:
    """Return a MegaCpp POC-style mask for tokens that should execute the layer."""
    batch, seq_len = router_logits.shape
    keep = mod_capacity(seq_len, capacity_factor)
    indices = mod_topk_indices(router_logits, keep)
    mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=router_logits.device)
    mask.scatter_(1, indices, True)
    return mask
