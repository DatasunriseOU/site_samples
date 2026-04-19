"""Public-safe ReDo dormancy-monitor example."""

from __future__ import annotations


def dormant_units(activity_by_unit: dict[str, float], threshold: float = 0.0) -> list[str]:
    return [name for name, value in activity_by_unit.items() if value <= threshold]


def redo_summary(activity_by_unit: dict[str, float], *, activation_family: str, threshold: float = 0.0) -> dict[str, object]:
    return {
        "activation_family": activation_family,
        "redo_enabled": activation_family in {"relu2", "relu"},
        "dormant_units": dormant_units(activity_by_unit, threshold=threshold),
    }
