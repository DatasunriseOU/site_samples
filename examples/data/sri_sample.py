"""Search-and-replace infill example.

This example converts a masked span into explicit SEARCH/REPLACE style text.
The point is to train the model on edit-shaped tasks, not only on plain hole
filling, so it learns to rewrite existing code with local context.
"""

from __future__ import annotations

import random


class SRIFormatter:
    def __init__(self, *, context_lines: int = 0, include_suffix_in_search: bool = True) -> None:
        self.context_lines = context_lines
        self.include_suffix_in_search = include_suffix_in_search

    def from_fim(self, prefix: str, suffix: str, middle: str) -> str:
        prefix_lines = prefix.splitlines()
        if self.context_lines > 0:
            prefix_lines = prefix_lines[-self.context_lines :]
        search_block = "\n".join(prefix_lines)
        if self.include_suffix_in_search and suffix:
            search_block = f"{search_block}\n{suffix}" if search_block else suffix
        return f"<<<<<<< SEARCH\n{search_block}\n=======\n{middle}\n>>>>>>> REPLACE"


def apply_sri(
    token_ids: list[int],
    tokenizer,
    *,
    sri_rate: float = 0.5,
    context_lines: int = 0,
    include_suffix: bool = True,
    rng: random.Random | None = None,
) -> list[int]:
    """Turn a token span into SEARCH/REPLACE edit format.

    This MegaCpp POC-style transform solves the mismatch between plain FIM and the
    way developers often express code changes: by editing an existing region.
    """

    active_rng = random.Random() if rng is None else rng
    if active_rng.random() > sri_rate or len(token_ids) < 2:
        return token_ids

    split_start = active_rng.randint(0, len(token_ids))
    split_end = active_rng.randint(split_start, len(token_ids))
    prefix_ids = token_ids[:split_start]
    middle_ids = token_ids[split_start:split_end]
    suffix_ids = token_ids[split_end:]

    prefix = tokenizer.decode(prefix_ids) if prefix_ids else ""
    middle = tokenizer.decode(middle_ids) if middle_ids else ""
    suffix = tokenizer.decode(suffix_ids) if suffix_ids else ""

    formatter = SRIFormatter(context_lines=context_lines, include_suffix_in_search=include_suffix)
    return tokenizer.encode(formatter.from_fim(prefix, suffix, middle))
