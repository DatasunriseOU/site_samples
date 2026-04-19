"""Semantic Tube Prediction (STP) geodesic-loss sample.

This example shows the auxiliary loss that nudges hidden-state trajectories to
stay locally straight. The point is to discourage unnecessary curvature in the
representation path by sampling ordered `(s, r, t)` triples and penalizing the
angle between consecutive direction vectors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_stp_loss(
    hidden_states: torch.Tensor | list[torch.Tensor],
    n_spans: int = 1,
) -> torch.Tensor:
    """Compute STP geodesic loss.

    Args:
        hidden_states: `(B, T, D)` tensor for a single layer, or a list of
            `(B, T, D)` tensors for the multi-layer variant.
        n_spans: Number of random `(s, r, t)` triples per sample.

    Returns:
        Scalar loss in `[0, 2]`. `0` means locally straight trajectories.
    """
    if isinstance(hidden_states, (list, tuple)):
        if len(hidden_states) == 0:
            return torch.tensor(0.0)
        losses = [_stp_loss_single(h, n_spans) for h in hidden_states]
        total_loss = torch.stack(losses).sum()
        return total_loss / len(losses)
    return _stp_loss_single(hidden_states, n_spans)


def _stp_loss_single(h: torch.Tensor, n_spans: int = 1) -> torch.Tensor:
    """STP loss on a single layer's hidden states."""
    batch_size, seq_len, hidden_dim = h.shape
    if seq_len < 3:
        return h.new_zeros(())

    # Use the donor's static-shape-friendly sampling pattern: draw 3 base
    # positions, sort them, then add `[0, 1, 2]` to guarantee `s < r < t`.
    max_base = seq_len - 2
    total_cos = h.new_zeros(())

    for _ in range(n_spans):
        base = torch.randint(0, max_base, (batch_size, 3), device=h.device)
        base, _ = base.sort(dim=-1)
        offsets = torch.arange(3, device=h.device, dtype=base.dtype).unsqueeze(0)
        positions = (base + offsets).clamp(max=seq_len - 1)

        idx_s = positions[:, 0].unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, hidden_dim)
        idx_r = positions[:, 1].unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, hidden_dim)
        idx_t = positions[:, 2].unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, hidden_dim)

        h_s = h.gather(1, idx_s).squeeze(1)
        h_r = h.gather(1, idx_r).squeeze(1)
        h_t = h.gather(1, idx_t).squeeze(1)

        d1 = h_r - h_s
        d2 = h_t - h_r
        cos_sim = F.cosine_similarity(d1, d2, dim=-1)
        total_cos = total_cos + cos_sim.mean()

    return 1.0 - total_cos / n_spans
