"""DualPipe stage-contract sample.

What it is: a public-safe summary of the stage-count and aux-loss rules used by
the MegaCpp POC DualPipe and DualPipeV helpers.

Why it exists: overlapped pipeline schedules break easily if stage counts,
output lifetime, or aux-loss injection are inconsistent.

What problem it solves: it spells out the invariants that keep forward/backward
overlap and router losses alive on non-terminal stages.
"""

from __future__ import annotations


def dualpipev_expected_stage_count(pp_degree: int) -> int:
    """Return the grounded total-stage count for DualPipeV."""

    if pp_degree < 1:
        raise ValueError("pp_degree must be positive")
    return 2 * pp_degree


def validate_dualpipev_boundaries(*, pp_degree: int, explicit_stage_count: int) -> str:
    """Mirror the MegaCpp POC explicit-boundary check for DualPipeV."""

    total_stages = dualpipev_expected_stage_count(pp_degree)
    if explicit_stage_count != total_stages:
        raise ValueError(
            f"DualPipeV with pp_degree={pp_degree} requires {total_stages} stages, got {explicit_stage_count}"
        )
    return "explicit boundaries match DualPipeV stage count"


def dualpipe_stage_runtime_rules() -> tuple[str, ...]:
    """Describe the key runtime rules around stage output lifetime and aux losses."""

    return (
        "do not deallocate stage outputs early because backward may need them after the next forward starts",
        "inject MoE and MoD auxiliary losses on non-terminal stages",
        "only the stage with logits owns the terminal criterion",
        "pipeline shape configuration must describe hidden-state tensors between stages",
    )
