"""Core FIRE/DASH/Redo donor excerpt used by public MegaCpp docs.

This excerpt intentionally keeps only the helper predicates plus the main
Newton-Schulz orthogonalization entrypoint. The full donor module also contains
compiled ReDo surgery kernels, optimizer resets, and dormancy recycling.
"""
import logging
from typing import Protocol, cast

import torch

logger = logging.getLogger(__name__)


class _ToLocalTensor(Protocol):
    def to_local(self) -> torch.Tensor: ...


class _TensorHolder(Protocol):
    _local_tensor: torch.Tensor


def _is_xla_tensor(t: torch.Tensor) -> bool:
    """Check if tensor lives on an XLA device."""
    return t.device.type == "xla"


def _local_tensor_if_dtensor(t: torch.Tensor) -> torch.Tensor:
    """Return the local tensor for a DTensor-like object, else the tensor itself."""
    for candidate in (t, getattr(t, "data", None)):
        if candidate is None:
            continue
        if hasattr(candidate, "_local_tensor"):
            return cast(_TensorHolder, candidate)._local_tensor
        to_local = getattr(candidate, "to_local", None)
        if callable(to_local):
            return cast(_ToLocalTensor, candidate).to_local()
    return t


def _is_dtensor_like(t: torch.Tensor) -> bool:
    """Best-effort DTensor-like predicate that works in tests and nightly wrappers."""
    try:
        from torch.distributed.tensor import DTensor as _DTensor

        return (
            isinstance(t, _DTensor)
            or hasattr(t, "_spec")
            or isinstance(getattr(t, "data", None), _DTensor)
            or hasattr(getattr(t, "data", None), "_spec")
        )
    except ImportError:
        return hasattr(t, "_spec") or hasattr(getattr(t, "data", None), "_spec")


@torch.no_grad()
def newton_schulz(W: torch.Tensor, iters: int = 15) -> torch.Tensor:
    """Approximate polar decomposition via Newton-Schulz iteration.

    Projects W onto the nearest orthogonal matrix (minimizes ||W - W_tilde||_F
    subject to W_tilde^T W_tilde = I). Preserves the original Frobenius norm
    to avoid breaking custom initialization scales.

    Args:
        W: 2D weight matrix (d_out, d_in)
        iters: Number of Newton-Schulz iterations (15 for reliable convergence;
               cubic iteration needs ~15 iters for condition numbers up to 500)

    Returns:
        Orthogonalized matrix with same Frobenius norm as input.
    """
    assert W.dim() == 2, f"newton_schulz requires 2D tensor, got {W.dim()}D"

    # Work in float32 for numerical stability (bf16 doesn't support linalg ops)
    orig_dtype = W.dtype
    W_f32 = W.float()

    # Normalize by spectral norm so all singular values are in (0, 1].
    # This ensures the cubic NS iteration converges (basin is (0, sqrt(3))).
    spectral_norm = torch.linalg.matrix_norm(W_f32, ord=2).clamp(min=1e-8)
    X = W_f32 / spectral_norm

    # Handle wide matrices (d_out < d_in): transpose so rows >= cols
    # Newton-Schulz converges only when the matrix has full column rank
    is_wide = W.shape[0] < W.shape[1]
    if is_wide:
        X = X.T

    # Standard Newton-Schulz coefficients (not Muon's tuned quintic)
    # These converge to true orthogonal matrix, not approximate
    a, b = 1.5, -0.5
    for _ in range(iters):
        A = X.T @ X
        X = a * X + b * (X @ A)

    if is_wide:
        X = X.T

    # Return the orthogonal matrix (all singular values ≈ 1.0).
    # Scaling for signal variance preservation is handled by apply_fire,
    # not here, because the correct scale depends on the model architecture.
    return X.to(orig_dtype)
