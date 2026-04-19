"""Instruction-aware FIM example.

This example adds a short instruction before the FIM span. The point is to
tell the model what kind of code should appear in the missing region instead of
making it infer the intent only from prefix and suffix.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


FIM_PREFIX_ID = 4
FIM_MIDDLE_ID = 5
FIM_SUFFIX_ID = 6
FIM_INSTRUCTION_ID = 7
EOT_ID = 3

_DOXYGEN_BRIEF_RE = re.compile(r"(?:@brief|\\brief)\s+(.+?)(?:\n\s*(?:@|\\\w|$)|\*/)", re.DOTALL)


@dataclass
class IFIMTokenIds:
    fim_prefix: int = FIM_PREFIX_ID
    fim_middle: int = FIM_MIDDLE_ID
    fim_suffix: int = FIM_SUFFIX_ID
    fim_instruction: int = FIM_INSTRUCTION_ID
    eot: int = EOT_ID


def extract_instruction_from_docstring(code: str) -> str | None:
    """Pull a short human instruction out of nearby documentation.

    This solves the common FIM problem where the hole is structurally correct
    but semantically underspecified.
    """

    match = _DOXYGEN_BRIEF_RE.search(code)
    if not match:
        return None
    text = " ".join(match.group(1).strip().split())
    return text or None


def apply_ifim(
    token_ids: list[int],
    *,
    text: str | None = None,
    tokenizer=None,
    ifim_rate: float = 0.5,
    instruction_rate: float = 0.5,
    spm_rate: float = 0.5,
    rng: random.Random | None = None,
) -> list[int]:
    active_rng = random.Random() if rng is None else rng
    if active_rng.random() >= ifim_rate:
        return token_ids
    if len(token_ids) < 2:
        return token_ids

    split_start = active_rng.randint(0, len(token_ids))
    split_end = active_rng.randint(split_start, len(token_ids))
    prefix = token_ids[:split_start]
    middle = token_ids[split_start:split_end]
    suffix = token_ids[split_end:]

    instruction_ids: list[int] = []
    if text and tokenizer is not None and active_rng.random() < instruction_rate:
        instruction = extract_instruction_from_docstring(text)
        if instruction:
            instruction_ids = [FIM_INSTRUCTION_ID] + list(tokenizer.encode(instruction))

    is_spm = active_rng.random() < spm_rate
    if is_spm:
        body = [FIM_PREFIX_ID, FIM_SUFFIX_ID] + suffix + [FIM_MIDDLE_ID] + prefix + middle + [EOT_ID]
    else:
        body = [FIM_PREFIX_ID] + prefix + [FIM_SUFFIX_ID] + suffix + [FIM_MIDDLE_ID] + middle + [EOT_ID]
    return instruction_ids + body
