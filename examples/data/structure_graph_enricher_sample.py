"""StructureGraphEnricher sample.

What it is: a public-safe excerpt of the TreeFFN-style chunk-graph enrichment
path used for structure-aware code training.

Why it exists: raw token streams do not preserve call edges, type dependency,
or chunk adjacency strongly enough on their own.

What problem it solves: it pools tokens to chunks, passes messages over the
real code graph, and scatters the enriched chunk state back to token positions
without changing the main token shape.
"""

from __future__ import annotations


EDGE_RELATION_INDICES = (1, 2, 3, 4, 5, 6)


def summarize_structure_graph_contract() -> dict[str, object]:
    return {
        "required_fields": {
            "chunk_relation_mask": "(B, R, C, C) bool relation tensor",
            "chunk_token_starts": "(B, C) chunk start positions",
            "chunk_token_ends": "(B, C) chunk end positions",
        },
        "optional_fast_path_fields": {
            "token_chunk_ids": "(B, T) precomputed token-to-chunk ids",
            "token_chunk_valid": "(B, T) validity mask for assigned chunks",
        },
        "message_passing": {
            "style": "TreeFFN-inspired chunk graph update",
            "relations_used": list(EDGE_RELATION_INDICES),
            "typical_edges": [
                "caller",
                "callee",
                "type_dep",
                "type_dependent",
                "same_dep_level",
                "adjacent_dep_level",
            ],
            "sparse_neighbor_cap": 16,
            "zero_init_output": True,
        },
        "runtime_effect": {
            "pool": "tokens -> chunks",
            "update": "graph message passing at chunk level",
            "scatter": "chunks -> tokens",
        },
    }
