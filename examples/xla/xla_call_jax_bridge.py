"""Splash `call_jax` bridge example for TPU attention.

This example shows the narrow bridge used when a TPU attention path still wants
to execute a JAX/Pallas kernel from a PyTorch training loop. It exists so the
rest of the model can stay in PyTorch while only the hot sparse-attention path
crosses the runtime boundary.

The MegaCpp POC path used this bridge for Splash-style kernels and then validated the
behavior with forward, backward, document-masking, and softcap checks.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch


def load_call_jax() -> Callable[..., Any] | None:
    """Load the TPU bridge when torch_xla exposes it."""

    try:
        xla_builder = importlib.import_module("torch_xla.core.xla_builder")
    except ImportError:
        return None
    return getattr(xla_builder, "call_jax", None)


def enable_splash_contract(attn_softcap: float) -> dict[str, object]:
    """Describe the runtime contract the MegaCpp POC tests exercised.

    The MegaCpp POC test suite re-enabled the bridge with multiple softcap values to
    make sure cache resets and kernel selection behaved predictably.
    """

    return {
        "enabled": True,
        "softcap": attn_softcap,
        "checks": (
            "forward",
            "backward",
            "segment_ids",
            "softcap_variants",
        ),
    }


def choose_bridge_path(*, device_type: str, call_jax_available: bool) -> str:
    """Select the narrow TPU bridge only when the runtime can support it."""

    if device_type == "xla" and call_jax_available:
        return "call_jax"
    if device_type == "xla":
        return "xla_fallback"
    return "non_xla_fallback"


def build_segment_ids(batch: int, tokens: int, split_at: int) -> torch.Tensor:
    """Recreate the MegaCpp POC document-mask setup used in the TPU bridge test."""

    segment_ids = torch.zeros(batch, tokens, dtype=torch.int32)
    segment_ids[:, split_at:] = 1
    return segment_ids


def bridge_summary() -> list[str]:
    return [
        "Keep PyTorch as the main frontend and use JAX only for the narrow TPU hotspot.",
        "Validate forward, backward, and document masking on the bridge path.",
        "Re-enable the bridge across several softcap values to catch cache or dispatch regressions.",
    ]
