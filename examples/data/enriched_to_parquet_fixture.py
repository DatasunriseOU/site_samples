"""Donor-grounded enriched JSONL fixture builder.

This mirrors the internal enriched-record example while keeping the sample
self-contained and public-safe.
"""

from __future__ import annotations

SAMPLE_TEXT = (
    "#include <vector>\n\n"
    "class Foo {\n"
    "public:\n"
    "  void bar();\n"
    "};\n\n"
    "void Foo::bar() {\n"
    "  std::vector<int> v;\n"
    "}\n"
)
SAMPLE_STRUCTURE_IDS = [1] * 19 + [4] * 20 + [5] * 17 + [2] * 11 + [3] * 32
SAMPLE_CHUNK_BOUNDARIES = [
    {"start": 0, "end": 19, "kind": 1, "dep_level": 0, "name": ""},
    {"start": 19, "end": 39, "kind": 4, "dep_level": 0, "name": "Foo"},
    {"start": 39, "end": 56, "kind": 5, "dep_level": 1, "name": "Foo::bar"},
    {"start": 56, "end": 67, "kind": 2, "dep_level": 0, "name": "Foo::bar"},
    {"start": 67, "end": 99, "kind": 3, "dep_level": 0, "name": "Foo::bar"},
]
SAMPLE_CALL_EDGES: list[dict[str, int]] = []
SAMPLE_TYPE_EDGES = [{"from": 0, "to": 1}]
SAMPLE_AST_DEPTH = [1] * 39 + [2] * 17 + [1] * 43
SAMPLE_SIBLING_INDEX = [0] * 39 + [1] * 17 + [0] * 43
SAMPLE_AST_NODE_TYPE = [10] * 19 + [20] * 20 + [30] * 17 + [40] * 11 + [50] * 32


def build_enriched_fixture() -> dict[str, object]:
    return {
        "text": SAMPLE_TEXT,
        "structure_ids": SAMPLE_STRUCTURE_IDS,
        "chunk_boundaries": SAMPLE_CHUNK_BOUNDARIES,
        "call_edges": SAMPLE_CALL_EDGES,
        "type_edges": SAMPLE_TYPE_EDGES,
        "ast_depth": SAMPLE_AST_DEPTH,
        "sibling_index": SAMPLE_SIBLING_INDEX,
        "ast_node_type": SAMPLE_AST_NODE_TYPE,
        "repo": "abseil",
        "filepath": "absl/sample/foo.cc",
        "commit_hash": "public-example-commit",
    }
