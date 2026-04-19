"""Compiled AdamW step sample.

This example shows how the fused optimizer step is compiled lazily once and
then reused unless runtime flags disable it. It exists because the inner
optimizer loop is hot enough that Python overhead matters, but debugging and
backend bring-up still need a clean eager fallback.

What problem it solves: it keeps the fast path compiled without making the
optimizer impossible to disable at runtime. The MegaCpp POC tests explicitly check
that late environment changes still prevent the first compile when needed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch

_adamw_step_compiled: Callable[..., Any] | None = None


def _adamw_step_fused_impl(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    lr_t: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2_t: torch.Tensor,
    eps_t: torch.Tensor,
    weight_decay_t: torch.Tensor,
) -> torch.Tensor:
    """Keep the fused update shape close to the MegaCpp POC contract."""
    _ = step_t
    if weight_decay_t.item() != 0:
        p.mul_(1 - lr_t * weight_decay_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.mul_(beta2_t).addcmul_(grad, grad, value=float(1 - beta2_t))
    denom = exp_avg_sq.sqrt().add_(eps_t)
    p.addcdiv_(exp_avg, denom, value=-float(lr_t))
    return p


def adamw_step_fused(*args, **kwargs):
    """Compile on first use unless compile is disabled before the first call."""
    global _adamw_step_compiled

    if os.environ.get("MEGACPP_NO_COMPILE", "0") == "1":
        return _adamw_step_fused_impl(*args, **kwargs)
    if os.environ.get("MEGACPP_NO_OPTIMIZER_COMPILE", "0") == "1":
        return _adamw_step_fused_impl(*args, **kwargs)

    if _adamw_step_compiled is None:
        try:
            _adamw_step_compiled = torch.compile(_adamw_step_fused_impl)
        except Exception:
            _adamw_step_compiled = _adamw_step_fused_impl
    return _adamw_step_compiled(*args, **kwargs)
