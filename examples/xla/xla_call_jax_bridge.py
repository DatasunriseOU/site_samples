"""call_jax TPU bridge example.

This shows how a TPU path can hand a hot kernel over to JAX/Pallas without
rewriting the whole model stack around JAX. The problem it solves is keeping
PyTorch as the main frontend while still reaching TPU-native kernels for a
small number of bottlenecks.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


def load_call_jax() -> Callable[..., Any] | None:
    try:
        xla_builder = importlib.import_module("torch_xla.core.xla_builder")
    except ImportError:
        return None
    return getattr(xla_builder, "call_jax", None)


def choose_bridge_path(*, device_type: str, zero_copy_available: bool) -> str:
    if device_type == "xla" and zero_copy_available:
        return "call_jax"
    return "fallback_roundtrip"


def bridge_summary() -> list[str]:
    return [
        "Keep PyTorch as the main training frontend.",
        "Use call_jax only for narrow TPU-native kernels that justify the bridge.",
        "Prefer zero-copy shared buffers on TPU when the runtime supports them.",
    ]
