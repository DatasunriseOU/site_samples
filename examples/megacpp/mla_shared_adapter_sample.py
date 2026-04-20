"""MegaCpp public example: shared MLA adapter seam.

What this solves in simple words:
- upstream attention layer specs drift;
- an MLA adapter can normalize only the MLA-specific parts while leaving the
  general builder untouched.
"""

from __future__ import annotations


def build_mla_shared_adapter() -> dict[str, object]:
    return {
        "adapter_name": "mla_shared",
        "handles_pp_layer_offset": True,
        "normalizes_rotary_behavior": True,
        "scope": "mla-specific compatibility only",
    }
