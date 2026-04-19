"""Pallas softcap attention normalization sample.

This example shows the small validity-prep layer that sits in front of the
TPU Pallas attention kernel. The point is to normalize token-validity metadata
into one contract so the kernel does not need many ad hoc mask formats.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionValidity:
    token_prefix: torch.Tensor | None = None
    base_block_tokens: int | None = None


def normalize_attention_validity(*, valid_token_counts: torch.Tensor, device: torch.device | str) -> AttentionValidity:
    return AttentionValidity(token_prefix=valid_token_counts.to(device), base_block_tokens=None)


def ensure_attention_validity(
    validity: AttentionValidity | None,
    *,
    query_tile_size: int,
    device: torch.device | str,
) -> AttentionValidity | None:
    if validity is None:
        return None
    if (
        validity.token_prefix is not None
        and validity.base_block_tokens is not None
        and validity.base_block_tokens != query_tile_size
    ):
        return normalize_attention_validity(valid_token_counts=validity.token_prefix, device=device)
    return validity
