"""Public-safe enriched-row to parquet-ready record example."""

from __future__ import annotations


def normalize_edges(edges: list[list[int]]) -> list[tuple[int, int]]:
    return [tuple(edge) for edge in edges]


def build_parquet_ready_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "repo": record["repo"],
        "filepath": record["filepath"],
        "language": record["language"],
        "text": record["text"],
        "chunk_boundaries": list(record.get("chunk_boundaries", [])),
        "structure_ids": list(record.get("structure_ids", [])),
        "call_edges": normalize_edges(list(record.get("call_edges", []))),
        "type_edges": normalize_edges(list(record.get("type_edges", []))),
    }
