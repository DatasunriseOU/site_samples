"""Public excerpt.

Source repo: MegaCpp public samples
Source material: https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/research/stp/stp-geodesic-regularizer__stp_loss_surface__v1.py
Purpose: show a minimal trajectory-straightness auxiliary loss sample used by the article
Edited for clarity.
"""

import torch
import torch.nn.functional as F


def _stp_loss_single(hidden_states: torch.Tensor, n_spans: int = 1) -> torch.Tensor:
    if hidden_states.size(1) < 3:
        return hidden_states.new_zeros(())

    total = hidden_states.new_zeros(())
    batch, steps, _ = hidden_states.shape

    for _ in range(n_spans):
        positions = torch.randint(0, max(1, steps - 2), (batch, 3), device=hidden_states.device)
        positions, _ = torch.sort(positions, dim=-1)
        positions = torch.clamp(positions + torch.arange(3, device=hidden_states.device), max=steps - 1)

        hs = hidden_states.gather(1, positions[:, 0].view(batch, 1, 1).expand(-1, 1, hidden_states.size(-1))).squeeze(1)
        hr = hidden_states.gather(1, positions[:, 1].view(batch, 1, 1).expand(-1, 1, hidden_states.size(-1))).squeeze(1)
        ht = hidden_states.gather(1, positions[:, 2].view(batch, 1, 1).expand(-1, 1, hidden_states.size(-1))).squeeze(1)

        d1 = hr - hs
        d2 = ht - hr
        total = total + (1.0 - F.cosine_similarity(d1, d2, dim=-1).mean())

    return total / n_spans


def compute_stp_loss(hidden_states, n_spans: int = 1) -> torch.Tensor:
    if isinstance(hidden_states, (list, tuple)):
        if not hidden_states:
            return torch.zeros(())
        losses = [_stp_loss_single(layer, n_spans=n_spans) for layer in hidden_states]
        return torch.stack(losses).mean()

    return _stp_loss_single(hidden_states, n_spans=n_spans)
