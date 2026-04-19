"""call_jax TPU bridge runtime sample.

What it is: a donor-based public-safe excerpt of the narrow bridge that hands a
TPU attention kernel from PyTorch into JAX/Pallas.
Why it exists: the model stayed in PyTorch, but a few TPU-native kernels were
worth bridging instead of reimplementing the full training stack in JAX.
What problem it solves: it keeps `call_jax` isolated to the exact hot path,
with a clear TPU zero-copy route and a separate fallback route for non-TPU
environments.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Protocol, cast

import torch


class XlaModelApi(Protocol):
    def mark_step(self) -> None: ...

    def xla_device(self) -> torch.device: ...


def prepare_tpu_runtime_env() -> None:
    """Mirror the narrow runtime flags used by the TPU call_jax smoke lane."""
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    logging.getLogger("absl").setLevel(logging.ERROR)
    os.environ.setdefault("XLA_NO_SPECIAL_SCALARS", "1")
    os.environ.setdefault("PJRT_DEVICE", "TPU")


def get_xla_model() -> XlaModelApi:
    return cast(XlaModelApi, importlib.import_module("torch_xla.core.xla_model"))


def load_call_jax():
    try:
        xla_builder = importlib.import_module("torch_xla.core.xla_builder")
    except ImportError:
        return None
    return getattr(xla_builder, "call_jax", None)


def choose_bridge_path(*, device: torch.device, call_jax_fn) -> str:
    if device.type == "xla" and call_jax_fn is not None:
        return "tpu_shared_buffer_bridge"
    return "fallback_copy_bridge"


def summarize_bridge_scope() -> tuple[str, str, str]:
    return (
        "Keep PyTorch as the main model frontend.",
        "Use call_jax only for narrow TPU-native kernels.",
        "Fall back cleanly when torch_xla or JAX is unavailable.",
    )


def example_device_probe() -> torch.device:
    prepare_tpu_runtime_env()
    xm = get_xla_model()
    return xm.xla_device()

