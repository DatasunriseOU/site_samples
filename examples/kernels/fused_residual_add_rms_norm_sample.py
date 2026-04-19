"""Fused residual-add plus RMSNorm contract.

What it is: the residual update plus RMSNorm pair that fires at block
boundaries throughout the model.

Why it exists: separate add and norm kernels repeat memory traffic and double
the launch count on one of the hottest paths in the runtime.

What problem it solves: one contract decides when the CUDA fused path is safe,
and otherwise preserves the exact eager semantics.
"""

from __future__ import annotations

import torch


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps).to(x.dtype)
    return x * rrms * weight


def fused_residual_add_rms_norm(
    residual: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
    weight: torch.Tensor,
    eps: float = 1e-8,
    *,
    fp32_residual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public sample of the fused residual+norm surface from `fused_residual.py`.

    Grounding:
    - MegaCpp POC source module: `fused_residual.py::fused_residual_add_rms_norm`
    - runtime use: transformer and hybrid block boundaries

    The important contract is the fast-path guard. The real Triton kernel only
    runs when all tensors are CUDA, contiguous, shape-aligned, and dtype-aligned.
    """
    fast_path = (
        residual.is_cuda
        and delta.is_cuda
        and weight.is_cuda
        and residual.is_contiguous()
        and delta.is_contiguous()
        and weight.is_contiguous()
        and residual.shape == delta.shape
        and residual.dtype == delta.dtype
        and weight.shape == (residual.shape[-1],)
    )

    if fp32_residual:
        new_residual = residual.to(torch.float32) + scale * delta.to(torch.float32)
        norm_input = new_residual.to(residual.dtype)
    else:
        new_residual = residual + scale * delta
        norm_input = new_residual

    normed = _rms_norm(norm_input, weight.to(norm_input.dtype), eps)
    if fast_path:
        return new_residual.to(residual.dtype), normed
    return new_residual.to(residual.dtype), normed
