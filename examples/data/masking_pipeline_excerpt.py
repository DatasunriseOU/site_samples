"""Compact masking-stage excerpt used by public data-processing docs."""

from __future__ import annotations


def normalize_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "repo": record.get("repo", "datasunriseou/megacpp-public"),
        "filepath": record.get("filepath", "src/example.cpp"),
        "language": record.get("language", "c++"),
        "text": str(record.get("text", "")),
        "chunk_boundaries": list(record.get("chunk_boundaries", [])),
        "structure_ids": list(record.get("structure_ids", [])),
        "call_edges": list(record.get("call_edges", [])),
        "type_edges": list(record.get("type_edges", [])),
    }


def apply_doc_masking(text: str, *, doc_open: str = "<doc>", doc_close: str = "</doc>") -> str:
    return f"{doc_open}{text.strip()}{doc_close}"


def build_enriched_row(record: dict[str, object]) -> dict[str, object]:
    normalized = normalize_record(record)
    return {
        "repo": normalized["repo"],
        "filepath": normalized["filepath"],
        "language": normalized["language"],
        "masked_text": apply_doc_masking(str(normalized["text"])),
        "chunk_boundaries": normalized["chunk_boundaries"],
        "structure_ids": normalized["structure_ids"],
        "call_edges": normalized["call_edges"],
        "type_edges": normalized["type_edges"],
    }
