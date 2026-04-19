"""FSDP2 plus LoRA sync sample.

This example explains a common trap: if LoRA weights are inserted after FSDP2
wrapping, those tiny parameters are not automatically managed by FSDP. The fix
is to add explicit gradient synchronization for the late-added LoRA tensors.
"""

from __future__ import annotations


def lora_sync_mode(*, injected_before_fsdp: bool) -> str:
    """Return the donor-backed sync strategy for LoRA parameters."""
    if injected_before_fsdp:
        return "ignore_in_fully_shard"
    return "register_grad_hooks"


def fsdp2_wrap_order() -> tuple[str, ...]:
    """Canonical donor order for TP, FSDP2, and late LoRA injection."""
    return (
        "apply tensor parallel first",
        "wrap the model or per-block layers with FSDP2",
        "if LoRA is injected later, attach gradient all-reduce hooks for those params",
    )
