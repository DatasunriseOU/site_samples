"""DASH tensor-mode excerpt."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def dash_step(
    W: torch.Tensor,
    grad: torch.Tensor,
    *,
    alpha: float = 0.05,
    shrink_rate: float = 0.01,
) -> torch.Tensor:
    if W.dim() != 2:
        raise ValueError(f"dash_step requires 2D tensors, got {W.dim()}D")

    cos_sim = F.cosine_similarity(W, grad, dim=1)
    penalty = torch.clamp(cos_sim - alpha, min=0.0).unsqueeze(1)
    shrink_factor = torch.clamp(1.0 - shrink_rate * penalty, min=0.5, max=1.0)
    return W * shrink_factor
