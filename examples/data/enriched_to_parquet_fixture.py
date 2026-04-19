"""Enriched-record normalization example.

This shows how one enriched JSONL record is normalized before parquet writing.
The problem it solves is schema drift: upstream tools may omit optional fields,
but parquet writers need one stable record shape every time.
"""

from __future__ import annotations

import json


def _normalize_chunk_boundaries(raw_boundaries: list[dict] | None) -> list[dict[str, int | str]]:
    normalized: list[dict[str, int | str]] = []
    for chunk in raw_boundaries or []:
        normalized.append(
            {
                "start": int(chunk.get("start", 0)),
                "end": int(chunk.get("end", 0)),
                "kind": str(chunk.get("kind", "")),
                "name": str(chunk.get("name", "")),
                "dep_level": min(int(chunk.get("dep_level", 0)), 255),
            }
        )
    return normalized


def _normalize_edges(raw_edges: list[dict] | None) -> list[dict[str, int]]:
    return [{"from": int(edge.get("from", 0)), "to": int(edge.get("to", 0))} for edge in raw_edges or []]


def _decode_constituent_provenance(raw: dict) -> list[dict[str, str | None]]:
    items = raw.get("constituent_provenance")
    if items is None and raw.get("constituent_provenance_json"):
        try:
            items = json.loads(raw["constituent_provenance_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            items = None
    decoded: list[dict[str, str | None]] = []
    if not isinstance(items, list):
        return decoded
    for item in items:
        if not isinstance(item, dict):
            continue
        decoded.append(
            {
                "filepath": item.get("filepath"),
                "language_info": json.dumps(item["language_info"]) if item.get("language_info") is not None else None,
                "build_info": json.dumps(item["build_info"]) if item.get("build_info") is not None else None,
            }
        )
    return decoded


def normalize_enriched_record(raw: dict) -> dict[str, object]:
    """Normalize one enriched JSONL record into a parquet-ready dict."""
    return {
        "text": raw.get("text", ""),
        "actual_token_count": int(raw.get("actual_token_count", 0) or 0),
        "structure_ids": list(raw.get("structure_ids", [])),
        "chunk_boundaries": _normalize_chunk_boundaries(raw.get("chunk_boundaries")),
        "call_edges": _normalize_edges(raw.get("call_edges")),
        "type_edges": _normalize_edges(raw.get("type_edges")),
        "ast_depth": list(raw.get("ast_depth", [])),
        "sibling_index": list(raw.get("sibling_index", [])),
        "ast_node_type": list(raw.get("ast_node_type", [])),
        "platform_info": json.dumps(raw["platform_info"]) if raw.get("platform_info") else None,
        "language_info": json.dumps(raw["language_info"]) if raw.get("language_info") else None,
        "build_info": json.dumps(raw["build_info"]) if raw.get("build_info") else None,
        "constituent_provenance": _decode_constituent_provenance(raw),
        "constituent_provenance_json": raw.get("constituent_provenance_json"),
        "repo": raw.get("repo"),
        "filepath": raw.get("filepath"),
        "commit_hash": raw.get("commit_hash"),
    }
