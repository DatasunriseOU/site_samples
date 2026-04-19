"""Donor-grounded TP partition excerpt from MLA refresh helpers."""

from __future__ import annotations


def partition_mla_heads(num_heads: int, tp_size: int) -> dict[str, int]:
    """Mirror MultiLatentAttention._refresh_mla_dims()."""
    if tp_size <= 1:
        return {"num_heads": num_heads, "tp_size": tp_size}
    if num_heads % tp_size != 0:
        raise ValueError(f"MLA n_head={num_heads} not divisible by tp_size={tp_size}")
    return {
        "num_heads": num_heads // tp_size,
        "tp_size": tp_size,
    }


def partition_mla_qkv_projector(
    num_heads: int,
    num_kv_heads: int,
    tp_size: int,
) -> dict[str, int]:
    """Mirror MLAQKVProjector._refresh_mla_dims()."""
    if tp_size <= 1:
        return {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "tp_size": tp_size,
        }
    if num_heads % tp_size != 0:
        raise ValueError(f"MLAQKVProjector n_head={num_heads} not divisible by tp_size={tp_size}")
    if num_kv_heads % tp_size != 0:
        raise ValueError(
            f"MLAQKVProjector n_kv_head={num_kv_heads} not divisible by tp_size={tp_size}"
        )
    return {
        "num_heads": num_heads // tp_size,
        "num_kv_heads": num_kv_heads // tp_size,
        "tp_size": tp_size,
    }
