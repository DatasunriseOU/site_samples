"""MegaCpp public example: wrap MLA integration behind a narrow compatibility seam.

What this solves in simple words:
- upstream attention specs drift over time;
- a small adapter layer is safer than scattering MLA-specific conditions through
  the whole model builder.
"""

from __future__ import annotations


def build_mla_adapter_config() -> dict[str, object]:
    return {
        "adapter": "mla_shared",
        "pp_layer_offset_safe": True,
        "rotary_pos_emb_mode": "normalized",
        "why_it_exists": "keep MLA-specific compatibility isolated from the general attention builder",
    }
