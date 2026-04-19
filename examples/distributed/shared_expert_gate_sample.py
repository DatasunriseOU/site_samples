"""Shared-expert gate sample.

This example shows the simple learned gate used on the always-on shared expert
path. The problem it solves is over-dominance: without a gate, the shared path
can wash out the routed specialists.
"""

from __future__ import annotations

import torch


def apply_shared_expert_gate(shared_out: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """Mirror the MegaCpp POC shared-expert sigmoid gate."""
    if shared_out.shape != gate_logits.shape:
        raise ValueError("shared_out and gate_logits must have the same shape")
    return shared_out * torch.sigmoid(gate_logits)


def combine_shared_and_routed(shared_out: torch.Tensor, routed_out: torch.Tensor) -> torch.Tensor:
    """Final MoE merge: shared path keeps the general behavior, routed path adds specialists."""
    if shared_out.shape != routed_out.shape:
        raise ValueError("shared_out and routed_out must have the same shape")
    return shared_out + routed_out
