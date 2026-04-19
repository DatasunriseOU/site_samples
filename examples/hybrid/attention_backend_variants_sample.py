"""Attention backend variant surface grounded in the MegaCpp POC sparse-attention config.

What it is: a small public-safe receipt for the backend and indexer choices that
drive sparse attention routing.

Why it exists: the same layer can run with dense SDPA-style sparse scoring,
chunked sparse kernels, or flex-style experimentation, and each backend has its
own helper options.

What problem it solves: it shows which knobs actually matter when changing an
attention backend, instead of mixing backend, indexer, and masking choices into
one vague preset label.
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_DSA_BACKENDS = frozenset({"sdpa", "chunked", "flex"})
SUPPORTED_CHUNKED_IMPLS = frozenset({"gather", "union", "fa3_gather", "fa3_gather_full"})


@dataclass(frozen=True)
class AttentionBackendChoice:
    dsa_backend: str
    dsa_chunked_impl: str
    dsa_indexer: str
    dsa_query_chunk_size: int
    dsa_local_window: int
    attn_softcap: float


def normalize_attention_backend_choice(config: object) -> AttentionBackendChoice:
    backend = str(getattr(config, "dsa_backend", "sdpa"))
    chunked_impl = str(getattr(config, "dsa_chunked_impl", "gather"))
    indexer = str(getattr(config, "dsa_indexer", "topk"))
    query_chunk = int(getattr(config, "dsa_query_chunk_size", 64))
    local_window = int(getattr(config, "dsa_local_window", 128))
    attn_softcap = float(getattr(config, "attn_softcap", 0.0))

    if backend not in SUPPORTED_DSA_BACKENDS:
        raise ValueError(f"unsupported dsa_backend {backend!r}")
    if chunked_impl not in SUPPORTED_CHUNKED_IMPLS:
        raise ValueError(f"unsupported dsa_chunked_impl {chunked_impl!r}")
    if query_chunk <= 0:
        raise ValueError("dsa_query_chunk_size must be > 0")
    if local_window < 0:
        raise ValueError("dsa_local_window must be >= 0")
    if attn_softcap < 0.0:
        raise ValueError("attn_softcap must be >= 0")

    return AttentionBackendChoice(
        dsa_backend=backend,
        dsa_chunked_impl=chunked_impl,
        dsa_indexer=indexer,
        dsa_query_chunk_size=query_chunk,
        dsa_local_window=local_window,
        attn_softcap=attn_softcap,
    )


def describe_attention_backend(config: object) -> dict[str, object]:
    choice = normalize_attention_backend_choice(config)
    return {
        "backend": choice.dsa_backend,
        "chunked_impl": choice.dsa_chunked_impl,
        "indexer": choice.dsa_indexer,
        "query_chunk_size": choice.dsa_query_chunk_size,
        "local_window": choice.dsa_local_window,
        "attn_softcap": choice.attn_softcap,
        "notes": {
            "sdpa": "dense math on a sparse token set",
            "chunked": "block-wise sparse kernels with explicit helper variants",
            "flex": "experimental sparse backend kept behind the same config surface",
        },
    }

