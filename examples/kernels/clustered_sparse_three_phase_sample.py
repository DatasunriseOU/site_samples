"""Clustered sparse three-phase kernel stage sample.

What it is: a public-safe receipt of the MegaCpp POC clustered sparse TPU path
that runs importance scoring, union selection, and sparse attention as three
separate stages.

Why it exists: the clustered sparse runtime has to keep routing stages
non-differentiable while still letting the final sparse attention stage receive
gradients.

What problem it solves: it makes the stage boundaries explicit so the runtime
does not silently mix routing logic with the differentiable attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClusteredSparseThreePhasePlan:
    use_fused_scoring: bool
    use_pallas: bool
    seq_len: int
    phase1: str
    phase2: str
    phase3: str
    routing_is_stop_gradient: bool
    attention_is_differentiable: bool


def build_clustered_sparse_three_phase_plan(
    *,
    use_fused_scoring: bool,
    use_pallas: bool,
    seq_len: int,
) -> ClusteredSparseThreePhasePlan:
    return ClusteredSparseThreePhasePlan(
        use_fused_scoring=use_fused_scoring,
        use_pallas=use_pallas,
        seq_len=seq_len,
        phase1=(
            "importance_scoring_pipeline_fused"
            if use_fused_scoring
            else "importance_scoring_pipeline"
        ),
        phase2="union_selection_pipeline",
        phase3="sparse_attention",
        routing_is_stop_gradient=True,
        attention_is_differentiable=True,
    )


def summarize_clustered_sparse_three_phase(
    *,
    use_fused_scoring: bool,
    use_pallas: bool,
    seq_len: int,
) -> dict[str, object]:
    plan = build_clustered_sparse_three_phase_plan(
        use_fused_scoring=use_fused_scoring,
        use_pallas=use_pallas,
        seq_len=seq_len,
    )
    return {
        "seq_len": plan.seq_len,
        "phase_order": [plan.phase1, plan.phase2, plan.phase3],
        "use_pallas": plan.use_pallas,
        "routing_is_stop_gradient": plan.routing_is_stop_gradient,
        "attention_is_differentiable": plan.attention_is_differentiable,
        "note": (
            "Phases 1 and 2 prepare sparse unions, while phase 3 runs the actual "
            "clustered sparse kernel."
        ),
    }
