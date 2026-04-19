"""GateSkip residual-budget sample.

What it is: a compact public-safe version of the residual gating policy used to
skip part of a block on low-importance tokens.

Why it exists: a layer-skip feature needs to start almost transparent and then
apply a controlled token budget instead of hard-dropping compute from step one.

What problem it solves: it shows how the MegaCpp POC keeps skip decisions tied
to a smooth gate and a scheduled budget, rather than an opaque boolean flag.
"""

from __future__ import annotations


def gateskip_budget(step: int, *, budget_start: float, budget_end: float, budget_warmup_steps: int) -> float:
    """Mirror the linear token-budget decay used by the MegaCpp POC GateSkip path."""

    if budget_warmup_steps <= 0:
        return budget_end
    clamped = max(0, min(step, budget_warmup_steps))
    alpha = clamped / budget_warmup_steps
    return (1.0 - alpha) * budget_start + alpha * budget_end


def gateskip_gate_init() -> dict[str, float]:
    """Return the grounded near-transparent init contract for GateSkip gates."""

    return {
        "weight_init_std": 0.01,
        "bias_init": 5.0,
        "why": "sigmoid(5) is near 1.0, so the model starts close to ungated",
    }


def gateskip_branch_contract() -> tuple[str, ...]:
    """Describe the public MegaCpp POC residual-gating contract."""

    return (
        "gate reads hidden states before the branch",
        "gate scales the branch output before adding back to the residual stream",
        "training uses smooth sigmoid gates with a sparsity penalty",
        "inference can harden the budget after training has shaped the scores",
    )
