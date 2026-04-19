"""Donor-backed activation-offload target mapping excerpt.

Adapted from the internal CPU activation offload module-name resolution
helpers. The example stays public-safe by describing generic transformer block
paths instead of any private run labels or environment identifiers.
"""

from __future__ import annotations


# Public-safe excerpt of the donor target vocabulary. The values mirror the
# real block-local attribute paths used when selective saved-tensor offload is
# attached to a transformer stack.
ACTIVATION_OFFLOAD_TARGETS: dict[str, list[str]] = {
    "qkv_linear": ["attn.c_qkv"],
    "attn_proj": ["attn.c_proj"],
    "core_attn": ["attn"],
    "mlp": ["mlp", "ffn"],
    "moe": ["ffn"],
    "expert_fc1": ["ffn"],
    "engram": ["engram"],
    "mhc": ["mhc"],
}


def activation_offload_targets() -> dict[str, list[str]]:
    """Return the donor-backed target-to-submodule mapping.

    The important contract from the donor is that offload is opt-in and
    targeted: callers enumerate specific block-local surfaces instead of trying
    to offload an entire model blindly.
    """

    return ACTIVATION_OFFLOAD_TARGETS.copy()
