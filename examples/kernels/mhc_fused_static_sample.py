"""mHC fused static primitive sample.

What it is: a public-safe excerpt of the static multi-stream mixing surface used
for the fused mHC CUDA fast path.

Why it exists: multi-stream residual mixing becomes expensive if each branch is
expanded and reduced by separate tiny kernels.

What problem it solves: it makes the static 4-stream mix contract explicit so a
single fused implementation can replace many small pointwise operations.
"""

from __future__ import annotations

import torch


def fused_stream_mix_static_sample(hidden: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Mirror the static mHC stream mix shape contract in pure Torch form."""

    if hidden.ndim != 4:
        raise ValueError(f"expected [B, T, S, D], got {tuple(hidden.shape)}")
    if hidden.shape[2] != 4:
        raise ValueError("static fused mHC sample expects 4 streams")
    if alpha.shape[-1] != hidden.shape[2]:
        raise ValueError("alpha stream dimension must match hidden stream count")

    weights = alpha.to(dtype=hidden.dtype).view(*alpha.shape[:-1], hidden.shape[2], 1)
    return (hidden * weights).sum(dim=2)
