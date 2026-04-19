"""Fused bias-dropout-add sample.

What it is: a public-safe excerpt of the compiled bias + dropout + residual-add
helper used at block boundaries.

Why it exists: plain elementwise chains are easy for the runtime to split into
separate kernels unless the helper keeps the contract explicit.

What problem it solves: it preserves one visible fusion surface for residual
updates so compiled training does not pay repeated launch overhead at every
block boundary.
"""

from __future__ import annotations

from typing import Optional

import torch


def bias_dropout_add_sample(
    x: torch.Tensor,
    *,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    """Mirror the MegaCpp POC bias-dropout-add contract in a readable form."""

    inplace = (
        not training
        and not x.requires_grad
        and not residual.requires_grad
        and (bias is None or not bias.requires_grad)
    )

    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)

    if bias is not None:
        x = x.add_(bias) if inplace else x + bias

    out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
    return out.add_(residual) if inplace else residual + out
