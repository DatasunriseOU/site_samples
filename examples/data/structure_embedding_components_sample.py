"""Structure embedding component sample.

What it is: a public-safe receipt for the token-level structure embedding inputs
used by the MegaCpp POC.

Why it exists: structure-aware training needs more than token IDs. It can add
structure category, dependency depth, AST depth, sibling index, and node type
signals at the input side.

What problem it solves: it makes the structure-input contract explicit so
enriched parquet fields stay aligned with the model features that consume them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureEmbeddingInputs:
    structure_ids: list[int]
    dep_levels: list[int]
    ast_depth: list[int]
    sibling_index: list[int]
    ast_node_type: list[int]


def summarize_structure_embedding_inputs(config: object) -> dict[str, object]:
    components = str(getattr(config, "structure_components", "core"))
    return {
        "components": components,
        "max_dep_level": int(getattr(config, "max_dep_level", 64)),
        "max_ast_depth": int(getattr(config, "max_ast_depth", 64)),
        "max_sibling_index": int(getattr(config, "max_sibling_index", 64)),
        "num_node_types": int(getattr(config, "num_node_types", 256)),
        "bottleneck_dim": int(getattr(config, "structure_bottleneck_dim", 64)),
        "uses": [
            "structure_ids",
            "dep_levels",
            "ast_depth",
            "sibling_index",
            "ast_node_type",
        ],
    }
