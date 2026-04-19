"""FSDP2 local-shard optimizer sample.

What it is: a compact public-safe excerpt of the shape checks the MegaCpp POC
optimizer uses when parameters are sharded by FSDP2.

Why it exists: optimizer code often sees full-shaped gradients and local-shaped
parameters at the same time.

What problem it solves: it shows how the local row shard is recovered before an
optimizer step, so the optimizer does not silently step the wrong view.
"""

from __future__ import annotations


def shard0_bounds(*, total_rows: int, world_size: int, rank: int) -> tuple[int, int]:
    """Match the first-dimension shard split used by the MegaCpp POC helper."""

    base = total_rows // world_size
    remainder = total_rows % world_size
    start = rank * base + min(rank, remainder)
    length = base + (1 if rank < remainder else 0)
    return start, length


def local_grad_slice(*, total_rows: int, local_rows: int, world_size: int, rank: int) -> dict[str, int]:
    """Return the grounded local-row slice for an FSDP2-style optimizer shard."""

    start, length = shard0_bounds(total_rows=total_rows, world_size=world_size, rank=rank)
    if length != local_rows:
        raise ValueError(
            f"local shard mismatch: expected local_rows={local_rows}, got shard length={length}"
        )
    return {
        "start_row": start,
        "row_count": length,
    }


def qkv_local_split_sizes(qkv_split_sizes: tuple[int, ...], *, local_row_dim: int) -> tuple[int, ...]:
    """Scale full-row split metadata down to the current local optimizer shard."""

    full_dim = sum(qkv_split_sizes)
    if full_dim <= 0 or local_row_dim <= 0 or local_row_dim == full_dim:
        return qkv_split_sizes
    scaled = tuple(part * local_row_dim // full_dim for part in qkv_split_sizes)
    remainder = local_row_dim - sum(scaled)
    if remainder:
        scaled = scaled[:-1] + (scaled[-1] + remainder,)
    return scaled
