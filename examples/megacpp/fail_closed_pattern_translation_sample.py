"""MegaCpp public example: translate a hybrid pattern fail-closed.

What this solves in simple words:
- a translator should refuse to silently reinterpret unsupported block families;
- the right public contract is to map what is supported and stop on the rest.
"""

from __future__ import annotations


SUPPORTED = {"A": "attention", "M": "mamba", "E": "moe", "D": "dense"}


def translate_pattern(pattern: str) -> list[str]:
    out = []
    for token in pattern:
        if token == "|":
            out.append("stage_break")
            continue
        if token not in SUPPORTED:
            raise ValueError(f"unsupported pattern token: {token}")
        out.append(SUPPORTED[token])
    return out
