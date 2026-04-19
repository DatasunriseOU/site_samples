"""TPU backend dispatch sample grounded in the MegaCpp POC TPU attention path.

What it is: a public-safe excerpt of the backend selection rules used when TPU
attention features need different kernel families.

Why it exists: Splash, Pallas, and chunked sparse helpers do not all cover the
same masking and softcap combinations.

What problem it solves: it documents the conservative dispatch order so new
features do not accidentally route into a backend that cannot honor the mask or
softcap contract.
"""

from __future__ import annotations


def choose_xla_attention_backend(
    *,
    causal: bool,
    sliding_window: bool,
    use_sparse_path: bool,
    use_modified_softcap: bool,
) -> str:
    """Mirror the practical TPU backend split from the MegaCpp POC stack."""

    if use_sparse_path:
        return "chunked_sparse_attention"
    if sliding_window or use_modified_softcap:
        return "xla_pallas"
    if causal:
        return "splash_attention"
    return "chunked_attention"


def generate_xla_backend_receipt() -> dict[str, object]:
    return {
        "device": "xla",
        "preferred_backends": [
            "splash_attention",
            "xla_pallas",
            "chunked_sparse_attention",
            "chunked_attention",
        ],
        "scenarios": {
            "causal_plain": choose_xla_attention_backend(
                causal=True,
                sliding_window=False,
                use_sparse_path=False,
                use_modified_softcap=False,
            ),
            "causal_softcap": choose_xla_attention_backend(
                causal=True,
                sliding_window=False,
                use_sparse_path=False,
                use_modified_softcap=True,
            ),
            "sliding_window": choose_xla_attention_backend(
                causal=True,
                sliding_window=True,
                use_sparse_path=False,
                use_modified_softcap=False,
            ),
            "sparse_path": choose_xla_attention_backend(
                causal=True,
                sliding_window=False,
                use_sparse_path=True,
                use_modified_softcap=False,
            ),
        },
    }

