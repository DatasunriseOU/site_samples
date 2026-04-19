"""Donor-based 3D parallelism planning excerpt.

Public-safe excerpt of the donor configuration contract and its main
divisibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParallelismConfig:
    """Configuration for 3D (PP x DP x TP x EP) parallelism.

    Args:
        pp: Pipeline parallelism degree.
        dp: Data parallelism degree.
        tp: Tensor parallelism degree.
        ep: Expert parallelism degree.
        pp_schedule: Pipeline schedule label.
        pp_microbatches: Number of microbatches used by pipeline parallelism.
    """

    pp: int = 1
    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp_schedule: str = "1f1b"
    pp_microbatches: int = 0

    def __post_init__(self) -> None:
        for name in ("pp", "dp", "tp", "ep"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.pp_microbatches < 0:
            raise ValueError(f"pp_microbatches must be >= 0, got {self.pp_microbatches}")

    @property
    def world_size(self) -> int:
        return self.pp * self.dp * self.tp * self.ep

    @property
    def has_pp(self) -> bool:
        return self.pp > 1

    @property
    def has_tp(self) -> bool:
        return self.tp > 1

    @property
    def has_ep(self) -> bool:
        return self.ep > 1

    @property
    def has_fsdp(self) -> bool:
        return self.dp > 1


def validate_3d_config(
    par: ParallelismConfig,
    model_config: Any,
    *,
    num_devices: int | None = None,
) -> list[str]:
    """Validate a 3D parallelism plan against a model config.

    Checks world-size agreement plus PP, TP, and EP divisibility.
    """

    warnings: list[str] = []
    n_layer = getattr(model_config, "n_layer", None) or getattr(model_config, "depth", None)
    n_head = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads", None)
    n_kv_head = getattr(model_config, "n_kv_head", None) or getattr(model_config, "num_query_groups", None)
    moe_experts = getattr(model_config, "moe_n_routed_experts", None) or getattr(model_config, "num_experts", None)

    if num_devices is not None and par.world_size != num_devices:
        raise ValueError(f"parallel world size {par.world_size} does not match available devices {num_devices}")

    if par.has_pp and n_layer is not None and n_layer % par.pp != 0:
        raise ValueError(f"PP requires layer count divisible by pp: n_layer={n_layer}, pp={par.pp}")

    if par.has_tp:
        if n_head is not None and n_head % par.tp != 0:
            raise ValueError(f"TP requires num_attention_heads divisible by tp: {n_head} vs {par.tp}")
        if n_kv_head is not None and n_kv_head % par.tp != 0:
            raise ValueError(f"TP requires kv heads divisible by tp: {n_kv_head} vs {par.tp}")

    if par.has_ep:
        if moe_experts is None:
            warnings.append("EP requested, but model config does not expose num_experts/moe_n_routed_experts")
        elif moe_experts % par.ep != 0:
            raise ValueError(f"EP requires expert count divisible by ep: {moe_experts} vs {par.ep}")

    if par.has_pp and par.pp_microbatches not in (0, 1) and par.pp_microbatches < par.pp:
        warnings.append("pp_microbatches lower than pp reduces pipeline overlap")

    return warnings


def describe_application_order() -> tuple[str, ...]:
    """Canonical order for applying the parallelism axes."""

    return (
        "TP splits weight matrices first",
        "PP partitions the layer stack into stages",
        "FSDP2 shards parameters inside each stage",
        "EP dispatches MoE tokens between expert peers after the stage layout is fixed",
    )
