"""XLA-safe AdamW optimizer — no .item(), no graph breaks, no recompilation.

Problem: torch.optim.AdamW calls `_get_value(step_t)` → `.item()` inside
`torch_xla.compile()`. This forces a graph break, and the subsequent graph
has different Python float constants (bias_correction, step_size) each step,
causing XLA to recompile from scratch (~48 min for NAM52).

Solution: Use .fill_() on pre-allocated 0-D device tensors (Muon's trick).
fill_() creates an IR node that, with XLA_NO_SPECIAL_SCALARS=1, hashes
identically regardless of the filled value. No .item(), no graph break.

capturable=True SEGFAULTs on TPU (libtpu crash in ConvertFromCppChunk).
"""

import torch
from torch.optim.optimizer import Optimizer


class XLAAdamW(Optimizer):
    """AdamW for XLA/TPU. All changing values use 0-D device tensors."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super().__init__(params, defaults)
        self._step_val = 0
        # 0-D tensors — will be moved to device on first step.
        # .fill_() changes value in-place without changing tensor identity.
        # With XLA_NO_SPECIAL_SCALARS=1, fill_(scalar) creates a DeviceData IR
        # node whose hash depends only on shape/dtype, NOT the scalar value.
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

            # Compute step-dependent values as Python floats (cheap, on CPU)
            bc1 = 1.0 - beta1 ** self._step_val
            bc2 = 1.0 - beta2 ** self._step_val
            step_size = lr / bc1
            bc2_sqrt = bc2 ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Move 0-D tensors to device once
                if not self._on_device:
                    self._bc2_sqrt_t = self._bc2_sqrt_t.to(p.device)
                    self._step_size_t = self._step_size_t.to(p.device)
                    self._wd_scale_t = self._wd_scale_t.to(p.device)
                    self._on_device = True

                # Fill 0-D tensors with current step values.
                # fill_() is an in-place op that does NOT call .item()
                # and does NOT break the XLA graph.
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

                # Decoupled weight decay — all tensor ops
                if wd != 0:
                    self._wd_scale_t.fill_(1.0 - lr * wd)
                    p.mul_(self._wd_scale_t)

                # Update moments (beta1/beta2 are constants, no recompilation)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Denominator
                if group["amsgrad"]:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / self._bc2_sqrt_t).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / self._bc2_sqrt_t).add_(eps)

                # Update parameters — ALL tensor ops, no Python scalars in graph
                # p = p - step_size * (exp_avg / denom)
                update = exp_avg / denom
                update.mul_(self._step_size_t)
                p.sub_(update)

        return loss
