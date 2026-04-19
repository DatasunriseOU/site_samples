"""Fused MLA down-projection, RMSNorm, and up-projection.

What it is: an experimental fused MLA helper that recomputes latent activations
instead of storing every intermediate in HBM.

Why it exists: MLA projection chains create large temporary tensors that are
expensive to save for backward.

What problem it solves: it shows the recomputation contract used to trade one
extra matmul in backward for lower activation residency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd import Function


class FusedDownNormUp(Function):
    """Public sample of the MLA projection autograd surface from `fused_mla_projection.py`."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_dkv: torch.Tensor,
        w_ukv: torch.Tensor,
        rms_weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        latent = F.linear(x, w_dkv)
        variance = latent.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rrms = torch.rsqrt(variance + eps).to(latent.dtype)
        normed = latent * rrms
        if rms_weight is not None:
            normed = normed * rms_weight
        output = F.linear(normed, w_ukv)
        ctx.save_for_backward(x, rrms, w_dkv, w_ukv, rms_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, rrms, w_dkv, w_ukv, rms_weight = ctx.saved_tensors
        latent = F.linear(x, w_dkv)
        normed = latent * rrms
        if rms_weight is not None:
            normed = normed * rms_weight

        grad_normed = F.linear(grad_output, w_ukv.t())
        grad_w_ukv = grad_output.reshape(-1, grad_output.shape[-1]).t() @ normed.reshape(-1, normed.shape[-1])
        grad_latent = grad_normed
        if rms_weight is not None:
            grad_latent = grad_latent * rms_weight
            grad_rms_weight = (grad_normed * (latent * rrms)).sum(dim=tuple(range(grad_normed.ndim - 1)))
        else:
            grad_rms_weight = None

        grad_x = F.linear(grad_latent * rrms, w_dkv.t())
        grad_w_dkv = (grad_latent * rrms).reshape(-1, grad_latent.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        return grad_x, grad_w_dkv, grad_w_ukv, grad_rms_weight, None


def fused_down_norm_up(
    x: torch.Tensor,
    w_dkv: torch.Tensor,
    w_ukv: torch.Tensor,
    rms_weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Apply the public MLA fused helper.

    Grounding:
    - MegaCpp POC source module: `fused_mla_projection.py`
    - runtime context: experimental MLA memory-reduction lane
    """
    return FusedDownNormUp.apply(x, w_dkv, w_ukv, rms_weight, eps)
