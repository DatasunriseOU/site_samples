"""MoDr router bookkeeping and auxiliary-loss sample.

What it is: a public-safe receipt for branch-router bookkeeping in MoDr,
including bias correction, optional auxiliary loss, and recurrence-step stats.

Why it exists: advanced routing needs more than branch IDs. The runtime tracks
balance, keeps per-step branch choices, and optionally adds a lightweight
router loss.

What problem it solves: it explains how MoDr avoids one branch taking all the
traffic while still keeping the main LM loss as the primary training signal.
"""

from __future__ import annotations


def build_modr_router_bookkeeping(config: object) -> dict[str, object]:
    """Summarize the grounded routing bookkeeping used in the MoDr loop."""

    aux_alpha = float(getattr(config, "aux_loss_alpha", 0.0))
    return {
        "router": {
            "num_branches": int(getattr(config, "num_branches", 3)),
            "topk": int(getattr(config, "num_branches_per_tok", 1)),
            "scoring": str(getattr(config, "scoring_func", "softmax")),
            "bias_correction": bool(getattr(config, "use_bias_correction", True)),
            "bias_update_rate": float(getattr(config, "bias_update_rate", 1e-4)),
            "aux_loss_alpha": aux_alpha,
        },
        "training_bookkeeping": {
            "records": [
                "branch_choices per recurrence step",
                "mean token-router scores per recurrence step",
                "running auxiliary loss across routed steps",
            ],
            "loss_merge": (
                "auxiliary branch-balance loss is averaged across routing steps and added to the LM loss"
                if aux_alpha > 0.0
                else "pure loss-free balancing mode keeps only bias correction without extra router loss"
            ),
        },
        "balance_story": {
            "loss_free_mode": "bias correction nudges underused branches upward without requiring a separate router objective",
            "with_aux_loss": "branch frequency and score mass can be multiplied into a small balancing penalty",
        },
    }
