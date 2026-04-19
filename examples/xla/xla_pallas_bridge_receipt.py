"""TPU Pallas bridge receipt sample.

This shows the checks used when a TPU attention path crosses into JAX-backed
Splash or Pallas helpers. It exists because the fast path is only useful when
forward, backward, document masking, and model integration all stay valid.
"""

from __future__ import annotations


def build_pallas_bridge_receipt() -> dict[str, object]:
    return {
        "device": "xla",
        "backend_chain": ["splash_attention", "xla_pallas", "chunked_attention"],
        "validated_cases": [
            "enable_splash_attention",
            "forward_bf16_attention",
            "backward_gradient_flow",
            "doc_ids_masking",
            "softcap_reconfiguration",
            "gpt_model_integration",
        ],
        "grounded_notes": [
            "The bridge is only considered healthy when forward and backward both complete without NaNs.",
            "Document masking must stay valid on the same runtime path, not on a separate fallback-only test.",
            "Softcap changes are re-applied through backend enablement to avoid stale cached configuration.",
        ],
    }
