"""Structure-aware FIM masking example.

This example shows how masking stays aligned to code structure instead of
cutting at arbitrary token positions. The point is to teach the model to fill
in function bodies or other coherent code spans without breaking the
surrounding metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


FIM_PREFIX_ID = 4
FIM_MIDDLE_ID = 5
FIM_SUFFIX_ID = 6
EOT_ID = 3


@dataclass
class FimResult:
    tokens: list[int]
    was_transformed: bool = False
    split_start: int = 0
    split_end: int = 0
    is_spm: bool = False


def apply_fim_function_level(
    token_ids: list[int],
    tokenizer,
    *,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    rng: random.Random | None = None,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
) -> FimResult:
    """Apply FIM around brace-delimited code blocks.

    This MegaCpp POC-style masking strategy prefers real function-like blocks over
    random spans, which makes the infill target look more like the code edits
    we want the model to learn.
    """

    active_rng = random.Random() if rng is None else rng
    if active_rng.random() >= fim_rate or len(token_ids) < 10:
        return FimResult(tokens=token_ids, was_transformed=False)

    try:
        open_brace_id = tokenizer.encode("{")[0]
        close_brace_id = tokenizer.encode("}")[0]
    except (IndexError, TypeError, AttributeError):
        return FimResult(tokens=token_ids, was_transformed=False)

    blocks: list[tuple[int, int]] = []
    depth = 0
    block_start = -1
    for index, token_id in enumerate(token_ids):
        if token_id == open_brace_id:
            if depth == 0:
                block_start = index
            depth += 1
        elif token_id == close_brace_id:
            depth -= 1
            if depth == 0 and block_start >= 0:
                body_len = index - block_start - 1
                if 4 <= body_len <= 500:
                    blocks.append((block_start, index))
                block_start = -1
            if depth < 0:
                depth = 0

    if not blocks:
        return FimResult(tokens=token_ids, was_transformed=False)

    split_start, split_end = active_rng.choice(blocks)
    prefix = token_ids[:split_start]
    middle = token_ids[split_start:split_end]
    suffix = token_ids[split_end:]
    is_spm = active_rng.random() < spm_rate

    if is_spm:
        result = [fim_prefix_id, fim_suffix_id] + suffix + [fim_middle_id] + prefix + middle + [eot_id]
    else:
        result = [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle + [eot_id]

    return FimResult(
        tokens=result,
        was_transformed=True,
        split_start=split_start,
        split_end=split_end,
        is_spm=is_spm,
    )
