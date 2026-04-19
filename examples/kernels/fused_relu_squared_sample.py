"""Fused ReLU-squared activation used in expert compute.

What it is: the relu(x)^2 activation surface used for routed experts and other
relu2 experiments.

Why it exists: eager `relu().square()` dispatches extra elementwise work and
saves more activation state than needed.

What problem it solves: the fused path keeps forward and backward small while
still falling back cleanly on non-CUDA devices.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.jit.script
def _fused_relu_squared_fwd(x: torch.Tensor) -> torch.Tensor:
    """Fuse `relu(x).square()` into one JIT-visible kernel surface."""
    return F.relu(x).square()


@torch.jit.script
def _fused_relu_squared_bwd(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Backward for relu^2 using the saved pre-activation input only."""
    return grad_output * (2.0 * F.relu(x))


class _FusedReLUSquaredFunction(torch.autograd.Function):
    """Autograd wrapper lifted from the MegaCpp POC relu2 helper."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return _fused_relu_squared_fwd(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        (x,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return (_fused_relu_squared_bwd(grad_output, x),)


def fused_relu_squared(x: torch.Tensor) -> torch.Tensor:
    """Public sample of `fused_relu_squared` from `fused_relu2.py`.

    Grounding:
    - MegaCpp POC source module: `fused_relu2.py`
    - runtime use: relu2 expert activations in MoE-oriented feature lanes
    """
    if x.is_cuda:
        out: torch.Tensor = _FusedReLUSquaredFunction.apply(x)  # type: ignore[assignment]
        return out
    return F.relu(x).square()
