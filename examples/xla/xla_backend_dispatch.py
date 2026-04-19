"""TPU backend dispatch example.

This shows how TPU attention backends are chosen for different masking modes.
The problem is that not every XLA backend supports every feature, so dispatch
must stay conservative and fall back cleanly when a fast path does not fit.
"""

from __future__ import annotations


def get_effective_xla_backend(*, causal: bool, sliding_window: bool) -> str:
    if not causal:
        return "chunked_attention"
    if sliding_window:
        return "xla_pallas"
    return "splash_attention"


def generate_xla_backend_receipt() -> dict[str, object]:
    return {
        "device": "xla",
        "preferred_backends": ["splash_attention", "xla_pallas", "chunked_attention"],
        "scenarios": {
            "causal_plain": get_effective_xla_backend(causal=True, sliding_window=False),
            "causal_sliding_window": get_effective_xla_backend(causal=True, sliding_window=True),
            "non_causal": get_effective_xla_backend(causal=False, sliding_window=False),
        },
    }
