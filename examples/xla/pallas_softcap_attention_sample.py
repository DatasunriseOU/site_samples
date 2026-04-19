"""TPU Pallas attention validity and loader surface.

This example shows the small helper layer that prepares runtime metadata before
the TPU Pallas attention kernels run. It exists so the fast path can accept one
stable mask and validity contract instead of many ad hoc call shapes.

The donor implementation added three important pieces:
- lazy TPU-only imports so CPU and CUDA code can still import shared helpers;
- validity normalization so tiled kernels see a consistent token-prefix view;
- dispatch rules that keep the native `trace_pallas` path for the modified
  softcap and mask-aware kernels.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch


class TracePallasFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]: ...


class FlashAttentionFn(Protocol):
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, /, **kwargs: Any) -> torch.Tensor: ...


@dataclass(frozen=True)
class AttentionValidity:
    token_prefix: torch.Tensor | None = None
    base_block_tokens: int | None = None


DEFAULT_MASK_VALUE = -0.7 * np.finfo(np.float32).max
MIN_BLOCK_SIZE = 128
NUM_LANES = 128
NUM_SUBLANES = 8


def require_trace_pallas() -> TracePallasFn:
    """Load the native TPU bridge only when the runtime actually needs it.

    The donor switched to `trace_pallas` so forward and backward stay on the
    XLA custom-kernel path instead of bouncing through a separate `call_jax`
    bridge.
    """

    trace_pallas = importlib.import_module("torch_xla.experimental.custom_kernel").trace_pallas
    return cast(TracePallasFn, trace_pallas)


def require_flash_attention() -> FlashAttentionFn:
    """Reach the unmodified TPU kernel for the simple no-softcap fallback."""

    flash_attention = importlib.import_module("torch_xla.experimental.custom_kernel").flash_attention
    return cast(FlashAttentionFn, flash_attention)


def normalize_attention_validity(*, valid_token_counts: torch.Tensor, device: torch.device | str) -> AttentionValidity:
    """Convert runtime token counts into the canonical attention-validity shape."""

    return AttentionValidity(token_prefix=valid_token_counts.to(device), base_block_tokens=None)


def ensure_attention_validity(
    validity: AttentionValidity | None,
    *,
    query_tile_size: int,
    device: torch.device | str,
) -> AttentionValidity | None:
    """Repair stale tile metadata so the kernel cannot silently use the wrong contract.

    In the donor path this keeps local-window and document-mask prep stable when
    tile sizes change across attention phases.
    """

    if validity is None:
        return None
    if (
        validity.token_prefix is not None
        and validity.base_block_tokens is not None
        and validity.base_block_tokens != query_tile_size
    ):
        return normalize_attention_validity(valid_token_counts=validity.token_prefix, device=device)
    return validity


def should_delegate_to_native_kernel(
    *,
    softcap: float,
    local_window: int,
    mask: object | None,
    causal: bool,
    gqa_groups: int,
) -> bool:
    """Mirror the donor dispatch rule for the simplest TPU path.

    When no added masking or softcap behavior is active, the donor falls back to
    the stock TPU flash-attention kernel rather than paying for the larger
    modified wrapper.
    """

    if softcap > 0.0:
        return False
    if local_window > 0:
        return False
    if mask is not None:
        return False
    if gqa_groups > 1:
        return False
    return True


def backend_summary() -> list[str]:
    return [
        "Forward and backward stay on the native trace_pallas path.",
        "Validity normalization keeps token-prefix metadata shape-stable across tiles.",
        "The simple no-softcap case can still reuse the stock TPU flash-attention kernel.",
    ]
