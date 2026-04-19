"""Regional compile block identity sample.

This example shows the compile rule used for distributed block compilation: try
the in-place ``module.compile(...)`` method first and only fall back if the
runtime truly does not expose it. It exists because distributed wrappers and
checkpoint routing depended on the original module identity staying visible.

What problem it solves: it avoids wrapping blocks in a new optimized module too
early, which can break later hook installation and make checkpoint policy drift
from the real compiled region.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def mark_block_compiled(block: nn.Module, *, compile_kwargs: dict[str, Any]) -> nn.Module:
    """Apply the MegaCpp POC identity-preserving compile rule to one block."""
    inner = getattr(block, "block", block)
    object.__setattr__(inner, "_skip_manual_checkpoint", True)
    try:
        block.compile(**compile_kwargs)
        return block
    except (AttributeError, TypeError):
        compiled = torch.compile(block, **compile_kwargs)
        return compiled


def identity_rule_notes() -> tuple[str, ...]:
    """Summarize why the MegaCpp POC prefers method-form regional compile."""
    return (
        "in-place compile preserves module identity for later distributed wrapping",
        "compiled blocks use inductor rematerialization instead of manual checkpoint wrappers",
        "fallback wrappers are only for runtimes that do not expose module.compile",
    )
