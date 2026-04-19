"""Chunked fused linear plus cross entropy.

What it is: the fallback-friendly chunked loss path that avoids materializing a
full `(tokens, vocab)` logits tensor at once.

Why it exists: even small specialist models hit large vocab costs, and the loss
path can dominate memory when logits are built monolithically.

What problem it solves: it trades a loop over chunks for bounded peak memory,
while keeping one contract across CUDA, CPU, and XLA.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def fused_linear_cross_entropy_chunked(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    softcap: Optional[float] = None,
    reduction: str = "mean",
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Public sample of the chunked loss contract from `kernels.py`.

    Grounding:
    - MegaCpp POC source module: `kernels.py::fused_linear_cross_entropy_chunked`
    - runtime use: bounded-memory loss path when fused vendor kernels are not
      the right fit for the active backend
    """
    if hidden_states.dim() == 3:
        batch_size, seq_len, width = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * seq_len, width)
        flat_targets = targets.reshape(batch_size * seq_len)
    else:
        flat_hidden = hidden_states
        flat_targets = targets.reshape(-1)

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    losses = []
    total = flat_hidden.shape[0]
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        logits = F.linear(flat_hidden[start:stop], lm_head_weight)
        if softcap is not None and softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        chunk_loss = F.cross_entropy(
            logits,
            flat_targets[start:stop],
            ignore_index=ignore_index,
            reduction="none",
        )
        losses.append(chunk_loss)

    merged = torch.cat(losses, dim=0) if losses else flat_hidden.new_empty(0)
    if reduction == "none":
        return merged
    if reduction == "sum":
        return merged.sum()
    valid = (flat_targets != ignore_index).sum().clamp_min(1)
    return merged.sum() / valid
