"""Residual path alternatives around mHC sample.

What it is: a public-safe summary of the residual-path switches that interact
with mHC in the MegaCpp POC.

Why it exists: mHC multi-stream mixing, fp32 residual accumulation, and AttnRes
all change how branch outputs re-enter the main stream.

What problem it solves: it makes the residual choices explicit so users can see
which combinations are safe and which ones compete for the same role.
"""

from __future__ import annotations


def describe_residual_paths(
    *,
    fp32_residual: bool,
    mhc_enabled: bool,
    mhc_n_streams: int,
    attn_res_enabled: bool,
) -> dict[str, object]:
    warnings: list[str] = []
    if attn_res_enabled and mhc_enabled and mhc_n_streams > 1:
        warnings.append(
            "AttnRes and multi-stream mHC both change cross-branch residual mixing; keep only one dominant path."
        )

    return {
        "fp32_residual": {
            "enabled": fp32_residual,
            "job": "perform residual accumulation in fp32 to reduce bf16 drift",
        },
        "mhc": {
            "enabled": mhc_enabled,
            "n_streams": mhc_n_streams,
            "job": "mix branch outputs with learned transport-style weights",
        },
        "attn_res": {
            "enabled": attn_res_enabled,
            "job": "add an extra attention-shaped residual correction path",
        },
        "warnings": warnings,
    }
