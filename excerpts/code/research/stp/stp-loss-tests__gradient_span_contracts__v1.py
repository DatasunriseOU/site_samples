"""Public excerpt.

Source repo: MegaCpp public samples
Purpose: show the minimal test contract used by STP article references
Edited for clarity.
"""

import torch

from stp_geodesic_regularizer__stp_loss_surface__v1 import compute_stp_loss


def test_stp_loss_handles_short_sequences() -> None:
    hidden = torch.randn(2, 2, 16, requires_grad=True)
    loss = compute_stp_loss(hidden, n_spans=2)
    assert loss.item() == 0.0


def test_stp_loss_propagates_gradients() -> None:
    hidden = torch.randn(2, 32, 64, requires_grad=True)
    loss = compute_stp_loss(hidden, n_spans=4)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_stp_loss_accepts_multiple_layers() -> None:
    layers = [torch.randn(2, 32, 64, requires_grad=True) for _ in range(4)]
    loss = torch.stack([compute_stp_loss(layer, n_spans=2) for layer in layers]).mean()
    loss.backward()
    assert all(layer.grad is not None for layer in layers)
