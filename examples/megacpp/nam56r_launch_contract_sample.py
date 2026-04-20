"""MegaCpp public example: split fixed launch policy from generated native args.

What this solves in simple words:
- some launch concerns are recipe-derived, others are cluster/runtime policy;
- keeping them separate makes changes auditable and safer to port.
"""

from __future__ import annotations


def build_launch_contract() -> dict[str, object]:
    return {
        "generated_native_args": [
            "--tensor-model-parallel-size=2",
            "--pipeline-model-parallel-size=4",
            "--sequence-parallel",
        ],
        "fixed_policy": [
            "set cache dirs before Python starts",
            "choose recipe mode before building native args",
            "keep graph policy as a launch concern",
        ],
    }
