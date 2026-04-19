"""MoD routing surface sample.

What it is: a public-safe summary of the real Mixture-of-Depths routing modes
used around hybrid blocks.

Why it exists: the MegaCpp POC supports several routing styles with different
runtime costs and different TPU/GPU tradeoffs.

What problem it solves: it makes the routing contract explicit so readers can
see when the model gathers tokens, when it uses threshold routing, and when it
keeps all tokens on the main path with a soft gate.
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_MOD_ROUTING = frozenset({"topk", "threshold", "gateskip"})


@dataclass(frozen=True)
class ModRoutingPlan:
    routing: str
    capacity: float
    threshold: float
    scorer: str
    target: str
    uses_gather_scatter: bool
    tpu_friendly: bool


def build_mod_routing_plan(config: object) -> ModRoutingPlan:
    routing = str(getattr(config, "mod_routing", "topk") or "topk")
    capacity = float(getattr(config, "mod_capacity", 0.5) or 0.0)
    threshold = float(getattr(config, "mod_threshold", 0.5) or 0.0)
    scorer = str(getattr(config, "mod_scorer", "learned") or "learned")
    target = str(getattr(config, "mod_target", "block") or "block")

    if routing not in SUPPORTED_MOD_ROUTING:
        raise ValueError(f"unsupported mod_routing {routing!r}")
    if not 0.0 <= capacity <= 1.0:
        raise ValueError("mod_capacity must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("mod_threshold must be in [0, 1]")

    uses_gather_scatter = routing in {"topk", "threshold"}
    tpu_friendly = routing in {"threshold", "gateskip"}

    return ModRoutingPlan(
        routing=routing,
        capacity=capacity,
        threshold=threshold,
        scorer=scorer,
        target=target,
        uses_gather_scatter=uses_gather_scatter,
        tpu_friendly=tpu_friendly,
    )


def describe_mod_surface(config: object) -> dict[str, object]:
    plan = build_mod_routing_plan(config)
    return {
        "routing": plan.routing,
        "capacity": plan.capacity,
        "threshold": plan.threshold,
        "scorer": plan.scorer,
        "target": plan.target,
        "uses_gather_scatter": plan.uses_gather_scatter,
        "tpu_friendly": plan.tpu_friendly,
        "notes": {
            "topk": "fixed compute budget with cross-token comparison",
            "threshold": "per-token routing without sorting, easier on XLA",
            "gateskip": "run the whole block and gate its contribution softly",
        },
    }
