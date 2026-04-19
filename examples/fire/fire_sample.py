"""Public-safe FIRE phase-boundary example."""

from __future__ import annotations


def select_fire_targets(named_params: dict[str, tuple[int, ...]], scope: str = "all") -> list[str]:
    touched = []
    for name, shape in named_params.items():
        if len(shape) != 2:
            continue
        if any(skip in name for skip in ("embed", "lm_head", "bias", "state")):
            continue
        if scope == "attention" and "attn" not in name:
            continue
        if scope == "mlp" and "mlp" not in name:
            continue
        touched.append(name)
    return touched


def fire_summary(named_params: dict[str, tuple[int, ...]], scope: str = "all") -> dict[str, object]:
    touched = select_fire_targets(named_params, scope=scope)
    return {"scope": scope, "fired_count": len(touched), "fired_names": touched}
