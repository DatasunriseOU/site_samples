"""Unified block choice sample grounded in the MegaCpp POC hybrid stack.

What it is: a public-safe excerpt of the logic that decides which parallel
attention-like branches live inside an A-block and how a superblock is built.

Why it exists: the model does not use one universal attention path. Standard
causal attention, sparse attention, Engram memory, and multi-stream mixing are
all optional surfaces, and the block constructor has to wire them together in a
predictable way.

What problem it solves: it makes the block menu explicit so readers can see
which features are enabled for a given layer instead of guessing from preset
names.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionBranchPlan:
    use_dense_attention: bool
    use_sparse_attention: bool
    use_engram: bool
    dense_gate_bias: float
    sparse_gate_bias: float
    engram_gate_bias: float


@dataclass(frozen=True)
class SuperBlockPlan:
    n_streams: int
    sinkhorn_iters: int
    dynamic_hyper_connections: bool
    dynamic_mode: str
    uses_fused_hc_ops: bool
    branch_plan: AttentionBranchPlan


def build_attention_branch_plan(*, use_engram: bool, use_dsa: bool) -> AttentionBranchPlan:
    """Mirror the warm-start gate policy from the hybrid A-block.

    The MegaCpp POC constructor always keeps dense attention present, always
    instantiates the sparse branch, and then biases the learned gates so the
    expected path dominates on day one.
    """

    return AttentionBranchPlan(
        use_dense_attention=True,
        use_sparse_attention=True,
        use_engram=use_engram,
        dense_gate_bias=2.0 if not use_dsa else -2.0,
        sparse_gate_bias=2.0 if use_dsa else -2.0,
        engram_gate_bias=2.0 if use_engram else -2.0,
    )


def build_superblock_plan(config: object, *, use_engram: bool, use_dsa: bool) -> SuperBlockPlan:
    """Summarize the hybrid superblock choices with the same config knobs.

    Grounded source surfaces:
    - `mhc_n_streams`
    - `mhc_sinkhorn_iters`
    - `mhc_dynamic`
    - `mhc_dynamic_mode`
    - `mhc_fused_ops`
    """

    return SuperBlockPlan(
        n_streams=int(getattr(config, "mhc_n_streams", 4)),
        sinkhorn_iters=int(getattr(config, "mhc_sinkhorn_iters", 5)),
        dynamic_hyper_connections=bool(getattr(config, "mhc_dynamic", False)),
        dynamic_mode=str(getattr(config, "mhc_dynamic_mode", "maxtext")),
        uses_fused_hc_ops=bool(getattr(config, "mhc_fused_ops", False)),
        branch_plan=build_attention_branch_plan(use_engram=use_engram, use_dsa=use_dsa),
    )


def summarize_ablock_choices(config: object, *, use_engram: bool, use_dsa: bool) -> dict[str, object]:
    """Return a compact public receipt of the grounded block-choice contract."""

    plan = build_superblock_plan(config, use_engram=use_engram, use_dsa=use_dsa)
    return {
        "superblock": {
            "n_streams": plan.n_streams,
            "sinkhorn_iters": plan.sinkhorn_iters,
            "dynamic_hyper_connections": plan.dynamic_hyper_connections,
            "dynamic_mode": plan.dynamic_mode,
            "uses_fused_hc_ops": plan.uses_fused_hc_ops,
        },
        "ablock": {
            "dense_attention": plan.branch_plan.use_dense_attention,
            "sparse_attention": plan.branch_plan.use_sparse_attention,
            "engram": plan.branch_plan.use_engram,
            "dense_gate_bias": plan.branch_plan.dense_gate_bias,
            "sparse_gate_bias": plan.branch_plan.sparse_gate_bias,
            "engram_gate_bias": plan.branch_plan.engram_gate_bias,
        },
    }
