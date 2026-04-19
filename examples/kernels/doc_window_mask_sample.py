"""Document-window mask builder sample.

What it is: a public-safe excerpt of the dense boolean mask builder that merges
causal order, document boundaries, valid-prefix counts, and an optional local
window.

Why it exists: several attention backends share the same logical mask rules but
need the rules materialized in one place.

What problem it solves: it prevents each backend wrapper from re-implementing
document masking slightly differently.
"""

from __future__ import annotations

import torch


def build_doc_window_mask_sample(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    valid_token_counts: torch.Tensor | None = None,
    doc_ids: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    row_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len, 1)
    col_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len)
    mask = col_idx <= row_idx

    left_window, right_window = window_size
    if left_window >= 0 and left_window < seq_len:
        mask = mask & ((row_idx - col_idx) <= left_window)
    if right_window >= 0 and right_window < seq_len:
        mask = mask & ((col_idx - row_idx) <= right_window)

    if valid_token_counts is not None:
        valid = valid_token_counts.to(device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        q_valid = row_idx < valid.view(batch_size, 1, 1)
        k_valid = col_idx < valid.view(batch_size, 1, 1)
        mask = mask & q_valid & k_valid

    if doc_ids is not None:
        docs = doc_ids.to(device=device, dtype=torch.long)
        mask = mask & docs.unsqueeze(2).eq(docs.unsqueeze(1))

    return mask.unsqueeze(1)
