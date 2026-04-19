"""Near-copy MegaCpp POC example: DSA index-cache patch seam.

This sample keeps the core contract visible: some DSA layers compute and cache
top-k indices, later layers reuse them, and a layer without a valid preceding
cache promotes itself back to a full path instead of silently running with an
undefined sparse-attention contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DsaLayerMode:
    rank: int
    mode: str
    cache_source: int | None
    notes: str


def full_layer(rank: int) -> DsaLayerMode:
    return DsaLayerMode(
        rank=rank,
        mode="full",
        cache_source=None,
        notes="compute sparse indices and refresh cache",
    )


def shared_layer(rank: int, cache_source: int) -> DsaLayerMode:
    return DsaLayerMode(
        rank=rank,
        mode="shared",
        cache_source=cache_source,
        notes="reuse nearest preceding full-layer cache",
    )


def promoted_layer(rank: int) -> DsaLayerMode:
    return DsaLayerMode(
        rank=rank,
        mode="promoted_to_full",
        cache_source=None,
        notes="no valid preceding cache on this stage, so the layer recomputes",
    )


def sample_index_cache_schedule() -> list[DsaLayerMode]:
    return [
        full_layer(0),
        shared_layer(1, 0),
        shared_layer(2, 0),
        full_layer(3),
        shared_layer(4, 3),
        promoted_layer(7),
    ]
