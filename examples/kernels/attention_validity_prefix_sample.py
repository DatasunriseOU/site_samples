"""Attention-validity prefix sample.

What it is: a public-safe excerpt of the prefix-validity normalization used by
attention wrappers.

Why it exists: sparse and dense attention helpers need one explicit contract
for valid token prefixes instead of many ad hoc kwargs.

What problem it solves: it keeps document-prefix and slot-prefix metadata in one
normalized shape so mask builders and backend selection can agree on what is
actually valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class AttentionValiditySample:
    mode: str
    token_prefix: torch.Tensor | None = None
    slot_counts: torch.Tensor | None = None
    base_block_tokens: int | None = None


def _coerce_counts(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    if tensor.ndim != 1:
        raise ValueError(f"expected rank-1 counts, got shape={tuple(tensor.shape)}")
    return tensor.to(dtype=torch.long)


def normalize_attention_validity_sample(
    *,
    attention_meta: Mapping[str, Any] | None = None,
    valid_token_counts: Any = None,
    valid_slot_counts: Any = None,
    base_block_tokens: Any = None,
) -> AttentionValiditySample:
    token_prefix = _coerce_counts(
        valid_token_counts
        if valid_token_counts is not None
        else (attention_meta or {}).get("row_valid_token_counts")
    )
    slot_counts = _coerce_counts(
        valid_slot_counts
        if valid_slot_counts is not None
        else (attention_meta or {}).get("row_valid_slot_counts")
    )
    base_tokens = (
        int(base_block_tokens)
        if base_block_tokens is not None
        else (attention_meta or {}).get("base_block_tokens")
    )

    if slot_counts is not None:
        if base_tokens is None or int(base_tokens) <= 0:
            raise ValueError("slot-prefix validity requires positive base_block_tokens")
        return AttentionValiditySample(
            mode="slot_prefix",
            token_prefix=token_prefix,
            slot_counts=slot_counts,
            base_block_tokens=int(base_tokens),
        )

    if token_prefix is not None:
        return AttentionValiditySample(mode="token_prefix", token_prefix=token_prefix)

    return AttentionValiditySample(mode="none")
