"""MegaCpp public example: add a recurrent-style mixer as a narrow Megatron seam.

What this solves in simple words:
- non-attention mixers should enter the stack through a small spec boundary,
  not through a large fork of the whole layer stack.
"""

from __future__ import annotations


def build_m2rnn_spec() -> dict[str, object]:
    return {
        "mixer": "m2rnn",
        "execution_role": "rblock",
        "narrow_seam": True,
        "why_it_exists": "keep recurrent-specific logic separate from the generic transformer path",
    }
