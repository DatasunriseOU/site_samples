"""MegaCpp public example: normalize extra structure inputs before embedding fusion.

What this solves in simple words:
- structure-aware embeddings need multiple aligned integer feature planes;
- the embedding seam should validate shape and dtype before mixing them into the
  main token embeddings.
"""

from __future__ import annotations

import torch


STRUCTURE_NAMES = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)


def normalize_structure_inputs(input_ids: torch.Tensor, structure_inputs: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if structure_inputs is None:
        raise ValueError("structure inputs are required for this example")
    normalized: dict[str, torch.Tensor] = {}
    for name in STRUCTURE_NAMES:
        tensor = structure_inputs.get(name)
        if tensor is None:
            continue
        if tensor.shape != input_ids.shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} must match input_ids {tuple(input_ids.shape)}")
        normalized[name] = tensor.to(device=input_ids.device, dtype=torch.long)
    return normalized
