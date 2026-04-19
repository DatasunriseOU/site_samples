"""Context-extension FIRE filter sample.

What it is: a small public-safe receipt for the narrow FIRE filter used during
context-extension transitions.

Why it exists: extending context is a phase change, but not every parameter
should be orthogonalized before the jump.

What problem it solves: it captures the topology-aware rule that keeps the
context-extension pass focused on attention Q/K-style surfaces and skips blocks
that are explicitly position-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FireTargetReceipt:
    block_name: str
    include_block: bool
    reason: str
    target_projection_names: tuple[str, ...]


def context_extension_fire_filter(*, block_name: str, is_mamba: bool, has_fused_qkv: bool) -> FireTargetReceipt:
    """Summarize the context-extension target filter used by FIRE."""

    if is_mamba:
        return FireTargetReceipt(
            block_name=block_name,
            include_block=False,
            reason="skip position-invariant block during context extension",
            target_projection_names=(),
        )
    if has_fused_qkv:
        return FireTargetReceipt(
            block_name=block_name,
            include_block=True,
            reason="fused attention projection carries the Q/K reset surface",
            target_projection_names=("c_qkv",),
        )
    return FireTargetReceipt(
        block_name=block_name,
        include_block=True,
        reason="legacy separate projections expose Q and K individually",
        target_projection_names=("c_q", "c_k"),
    )
