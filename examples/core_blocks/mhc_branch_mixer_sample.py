"""MegaCpp POC-based mHC branch mixer example.

This feature mixes multiple branch outputs with constrained weights instead of
just adding them blindly. The point is to let the model combine residual,
Engram, and other side branches without one branch dominating by accident.

The sample keeps the stable public-safe core of the MegaCpp POC implementation:
- pooled branch scoring;
- Sinkhorn normalization when branch count is larger than two;
- a safe fallback to softmax for the two-branch case.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ManifoldBranchMixerSample(nn.Module):
    """Constrained branch mixer with Sinkhorn-normalized weights."""

    def __init__(
        self,
        n_embd: int,
        sinkhorn_iters: int = 5,
        temperature: float = 1.0,
        epsilon: float = 1e-6,
        blend_alpha: float = 1.0,
        max_branches: int = 0,
    ) -> None:
        super().__init__()
        hidden = max(8, min(256, n_embd // 4))
        self.score_proj = nn.Linear(n_embd, hidden, bias=False)
        self.score_out = nn.Linear(hidden, 1, bias=False)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.blend_alpha = float(blend_alpha)
        self.max_branches = int(max_branches)

    def _sinkhorn(self, raw_matrix: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon
        transport = torch.softmax(raw_matrix.float(), dim=-2)
        for _ in range(max(0, self.sinkhorn_iters)):
            transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(eps)
            transport = transport / transport.sum(dim=-2, keepdim=True).clamp_min(eps)
        return transport

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        if not branches:
            raise ValueError("ManifoldBranchMixerSample requires at least one branch")
        if len(branches) == 1:
            return branches[0]
        if self.max_branches > 0 and len(branches) > self.max_branches:
            raise ValueError(f"Too many branches: got {len(branches)}, max_branches={self.max_branches}")

        ref_shape = branches[0].shape
        for branch in branches[1:]:
            if branch.shape != ref_shape:
                raise ValueError(f"Branch shape mismatch: expected {ref_shape}, got {branch.shape}")

        stacked = torch.stack(branches, dim=2)
        pooled = stacked.mean(dim=1)
        logits = self.score_out(torch.tanh(self.score_proj(pooled))).squeeze(-1)
        temperature = max(self.temperature, self.epsilon)
        n_branches = stacked.size(2)

        if n_branches == 2:
            weights = torch.softmax(logits / temperature, dim=-1)
        else:
            keys = self.score_proj(pooled)
            raw_matrix = torch.bmm(keys, keys.transpose(-2, -1)) / temperature
            raw_matrix = raw_matrix + torch.diag_embed(logits / temperature)
            transport = self._sinkhorn(raw_matrix.float())
            weights = torch.diagonal(transport, dim1=-2, dim2=-1)

        uniform = torch.full_like(weights, 1.0 / n_branches)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
        is_valid = torch.isfinite(weights).all(dim=-1, keepdim=True)
        weights = torch.where(is_valid, weights, uniform)

        alpha = min(max(self.blend_alpha, 0.0), 1.0)
        if alpha < 1.0:
            weights = uniform + alpha * (weights - uniform)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
        weights = weights.to(dtype=stacked.dtype)
        return (stacked * weights[:, None, :, None]).sum(dim=2)
