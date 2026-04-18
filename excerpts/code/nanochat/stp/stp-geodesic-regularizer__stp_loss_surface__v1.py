"""Sanitized public excerpt.

Source repo: MegaCpp research repo
Source file: stp module
Purpose: show the minimal STP loss surface used by the article
Edited for public clarity.
"""

import torch
import torch.nn.functional as F


def compute_stp_loss(hidden_states: torch.Tensor, n_spans: int = 1) -> torch.Tensor:
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
