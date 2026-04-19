"""Fixture builder for the enriched parquet donor examples.

This stays next to the real donor excerpts so article links can point to one
compact record shape without mixing in older toy pipeline code.
"""

from __future__ import annotations


def build_enriched_fixture() -> dict[str, object]:
    return {
        "text": "int sum(int a, int b) { return a + b; }",
        "structure_ids": [1, 1, 1, 2, 2, 3, 3],
        "chunk_boundaries": [
            {"start": 0, "end": 21, "kind": "signature", "name": "sum", "dep_level": 0},
            {"start": 22, "end": 40, "kind": "body", "name": "sum", "dep_level": 1},
        ],
        "call_edges": [],
        "type_edges": [{"from": 1, "to": 2}],
        "repo": "datasunriseou/megacpp-public",
        "filepath": "src/sum.cpp",
        "commit_hash": "public-example-commit",
    }
