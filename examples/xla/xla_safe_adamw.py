"""XLA-safe AdamW example.

This shows how to run AdamW on TPU without forcing graph breaks every step.
The problem is that pulling changing scalar values back to Python can trigger
recompilation under torch_xla. The fix is to keep the changing values in
device tensors and update them in place.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class XLAAdamW(Optimizer):
    """AdamW variant that keeps step-dependent scalars on device tensors."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)
        self._step_val = 0
        self._bc2_sqrt_t = torch.tensor(1.0)
        self._step_size_t = torch.tensor(0.0)
        self._wd_scale_t = torch.tensor(1.0)
        self._on_device = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_val += 1

        for group in self.param_groups:
            if not group["params"]:
                continue

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]

            bc1 = 1.0 - beta1**self._step_val
            bc2 = 1.0 - beta2**self._step_val
            step_size = lr / bc1
            bc2_sqrt = bc2**0.5

            for p in group["params"]:
                if p.grad is None:
                    continue

                if not self._on_device:
                    self._bc2_sqrt_t = self._bc2_sqrt_t.to(p.device)
                    self._step_size_t = self._step_size_t.to(p.device)
                    self._wd_scale_t = self._wd_scale_t.to(p.device)
                    self._on_device = True

                self._bc2_sqrt_t.fill_(bc2_sqrt)
                self._step_size_t.fill_(step_size)

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if wd != 0:
                    self._wd_scale_t.fill_(1.0 - lr * wd)
                    p.mul_(self._wd_scale_t)

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsgrad"]:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / self._bc2_sqrt_t).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / self._bc2_sqrt_t).add_(eps)

                update = exp_avg / denom
                update.mul_(self._step_size_t)
                p.sub_(update)

        return loss
