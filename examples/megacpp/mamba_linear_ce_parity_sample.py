"""MegaCpp POC example: keep Mamba-style output projection and CE loss parity explicit.

What this solves in simple words:
- when swapping or refactoring the output path, it is easy to keep tensor shapes
  compatible while drifting on the loss contract;
- a small parity sample keeps the logits->CE interface visible and testable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_ce_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = hidden @ weight.T
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def parity_check(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> float:
    ref = linear_ce_loss(hidden, weight, targets)
    alt = linear_ce_loss(hidden.clone(), weight.clone(), targets.clone())
    return float((ref - alt).abs())
