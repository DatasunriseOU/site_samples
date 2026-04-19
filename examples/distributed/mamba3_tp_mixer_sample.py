"""Grounded excerpt of the TP-aware Mamba3 mixer contract.

This keeps the MegaCpp POC topology arithmetic and the rationale around replicated vs
sharded parameters, but trims away the full Megatron module body and kernel
calls so the sample remains readable and import-clean.
"""

from __future__ import annotations

from dataclasses import dataclass


def _assert_divisible(name: str, value: int, tp_size: int) -> int:
    if value % tp_size != 0:
        raise ValueError(
            f"CppmegaMamba3TPMixer: {name}={value} is not evenly divisible by "
            f"tp_world_size={tp_size}; every TP rank must receive an integer number of heads/groups"
        )
    return value // tp_size


@dataclass(frozen=True)
class Mamba3StructuralConfig:
    d_model: int
    expand: int
    headdim: int
    ngroups: int
    d_state: int
    mimo_rank: int
    rope_fraction: float = 0.5


@dataclass(frozen=True)
class TPMixerLayout:
    tp_world_size: int
    nheads: int
    ngroups: int
    nheads_local_tp: int
    ngroups_local_tp: int
    d_inner: int
    d_inner_local_tp: int
    d_in_proj_full: int
    d_in_proj_local: int
    num_rope_angles: int
    angle_proj_is_replicated: bool


def build_tp_mixer_layout(config: Mamba3StructuralConfig, *, tp_world_size: int) -> TPMixerLayout:
    d_inner = config.d_model * config.expand
    if d_inner % config.headdim != 0:
        raise ValueError(f"d_inner={d_inner} must be divisible by headdim={config.headdim}")

    nheads = d_inner // config.headdim
    if nheads % config.ngroups != 0:
        raise ValueError(f"nheads={nheads} must be divisible by ngroups={config.ngroups}")

    nheads_local_tp = _assert_divisible("nheads", nheads, tp_world_size)
    ngroups_local_tp = _assert_divisible("ngroups", config.ngroups, tp_world_size)
    d_inner_local_tp = nheads_local_tp * config.headdim

    split_tensor_size = int(config.d_state * config.rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2
    if num_rope_angles <= 0:
        raise ValueError("num_rope_angles must be positive")

    d_in_proj_full = 2 * d_inner + 2 * config.ngroups * config.d_state * config.mimo_rank + 3 * nheads
    d_in_proj_local = d_in_proj_full // tp_world_size

    return TPMixerLayout(
        tp_world_size=tp_world_size,
        nheads=nheads,
        ngroups=config.ngroups,
        nheads_local_tp=nheads_local_tp,
        ngroups_local_tp=ngroups_local_tp,
        d_inner=d_inner,
        d_inner_local_tp=d_inner_local_tp,
        d_in_proj_full=d_in_proj_full,
        d_in_proj_local=d_in_proj_local,
        num_rope_angles=num_rope_angles,
        angle_proj_is_replicated=True,
    )


def explain_tp_contract() -> tuple[str, ...]:
    return (
        "in_proj is column-parallel and packs [z, x, B, C, dd_dt, dd_A, trap] across local heads/groups",
        "out_proj consumes the local d_inner shard and all-reduces back to full d_model",
        "angle_proj stays replicated across TP ranks so every head sees identical rotation angles",
        "per-head parameters live at nheads_local_tp and shard cleanly along partition_dim=0",
    )
