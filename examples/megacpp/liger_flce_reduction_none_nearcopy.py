"""Near-copy MegaCpp POC example: Liger FLCE `reduction='none'` contract.

This sample preserves the public shape of the real reproducer: one backward
path is wrong under `reduction='none'`, `reduction='mean'` stays valid, and a
scaled-mean workaround recovers per-token sum semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradientReceipt:
    reduction: str
    grad_hidden_ok: bool
    grad_weight_ok: bool
    has_nan: bool
    notes: str


def eager_reference_contract() -> GradientReceipt:
    return GradientReceipt(
        reduction="reference",
        grad_hidden_ok=True,
        grad_weight_ok=True,
        has_nan=False,
        notes="F.linear + cross_entropy eager baseline",
    )


def liger_reduction_none_contract() -> GradientReceipt:
    return GradientReceipt(
        reduction="none",
        grad_hidden_ok=False,
        grad_weight_ok=False,
        has_nan=True,
        notes="non-uniform grad_output path can corrupt chunked grad_weight accumulation",
    )


def liger_reduction_mean_contract() -> GradientReceipt:
    return GradientReceipt(
        reduction="mean",
        grad_hidden_ok=True,
        grad_weight_ok=True,
        has_nan=False,
        notes="known-working reduction path",
    )


def scaled_mean_workaround_contract() -> GradientReceipt:
    return GradientReceipt(
        reduction="mean * n_valid",
        grad_hidden_ok=True,
        grad_weight_ok=True,
        has_nan=False,
        notes="recovers per-token sum semantics through valid-token scaling",
    )


def compare_flce_paths() -> dict[str, GradientReceipt]:
    return {
        "reference": eager_reference_contract(),
        "liger_none": liger_reduction_none_contract(),
        "liger_mean": liger_reduction_mean_contract(),
        "scaled_mean_workaround": scaled_mean_workaround_contract(),
    }
