#!/usr/bin/env python3
"""Grounded donor excerpt for enriched JSONL -> parquet normalization.

This keeps the real record-normalization structure from the production data
pipeline, but trims away tokenizer-coupled materialization so the sample stays
public-safe and self-contained.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import SupportsIndex, SupportsInt, TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq

_ConvertibleToInt: TypeAlias = str | bytes | bytearray | SupportsInt | SupportsIndex

REPO_COLUMN = "repo"
FILEPATH_COLUMN = "filepath"
COMMIT_HASH_COLUMN = "commit_hash"
TIMESTAMP_COLUMN = "timestamp"
PARENT_HASHES_COLUMN = "parent_hashes"
PARENT_COUNT_COLUMN = "parent_count"
IS_MERGE_COMMIT_COLUMN = "is_merge_commit"
AUTHOR_TIMESTAMP_COLUMN = "author_timestamp"
COMMIT_TIMESTAMP_COLUMN = "commit_timestamp"
REPO_STABLE_ID_COLUMN = "repo_stable_id"
FILEPATH_STABLE_ID_COLUMN = "filepath_stable_id"
FILE_LOCAL_COMMIT_INDEX_COLUMN = "file_local_commit_index"
HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN = "has_ambiguous_reconstruction"
HAS_RENAME_AMBIGUITY_COLUMN = "has_rename_ambiguity"
CHANGED_CHUNK_IDS_COLUMN = "changed_chunk_ids"
CHANGED_CHUNK_SPANS_COLUMN = "changed_chunk_spans"
TOKEN_CHANGE_MASK_PRE_COLUMN = "token_change_mask_pre"
TOKEN_CHANGE_MASK_POST_COLUMN = "token_change_mask_post"
HUNK_ID_PER_TOKEN_COLUMN = "hunk_id_per_token"
EDIT_OP_PER_TOKEN_COLUMN = "edit_op_per_token"


def _optional_int(value: _ConvertibleToInt | None) -> int | None:
    if value is None:
        return None
    return int(value)


CHUNK_BOUNDARY_TYPE = pa.struct(
    [
        pa.field("start", pa.uint32()),
        pa.field("end", pa.uint32()),
        pa.field("kind", pa.string()),
        pa.field("name", pa.string()),
        pa.field("dep_level", pa.uint8()),
    ]
)

EDGE_TYPE = pa.struct([pa.field("from", pa.uint16()), pa.field("to", pa.uint16())])

CONSTITUENT_PROVENANCE_TYPE = pa.struct(
    [
        pa.field("filepath", pa.string()),
        pa.field("language_info", pa.string()),
        pa.field("build_info", pa.string()),
    ]
)

SCHEMA = pa.schema(
    [
        pa.field("text", pa.string()),
        pa.field("actual_token_count", pa.int32()),
        pa.field("structure_ids", pa.list_(pa.uint8())),
        pa.field("chunk_boundaries", pa.list_(CHUNK_BOUNDARY_TYPE)),
        pa.field("call_edges", pa.list_(EDGE_TYPE)),
        pa.field("type_edges", pa.list_(EDGE_TYPE)),
        pa.field("ast_depth", pa.list_(pa.uint8())),
        pa.field("sibling_index", pa.list_(pa.uint8())),
        pa.field("ast_node_type", pa.list_(pa.uint8())),
        pa.field("platform_info", pa.string()),
        pa.field("language_info", pa.string()),
        pa.field("build_info", pa.string()),
        pa.field("constituent_provenance", pa.list_(CONSTITUENT_PROVENANCE_TYPE)),
        pa.field("constituent_provenance_json", pa.string()),
        pa.field(REPO_COLUMN, pa.string()),
        pa.field(FILEPATH_COLUMN, pa.string()),
        pa.field(COMMIT_HASH_COLUMN, pa.string()),
        pa.field(TIMESTAMP_COLUMN, pa.string()),
        pa.field(PARENT_HASHES_COLUMN, pa.list_(pa.string())),
        pa.field(PARENT_COUNT_COLUMN, pa.int32()),
        pa.field(IS_MERGE_COMMIT_COLUMN, pa.bool_()),
        pa.field(AUTHOR_TIMESTAMP_COLUMN, pa.string()),
        pa.field(COMMIT_TIMESTAMP_COLUMN, pa.string()),
        pa.field(REPO_STABLE_ID_COLUMN, pa.string()),
        pa.field(FILEPATH_STABLE_ID_COLUMN, pa.string()),
        pa.field(FILE_LOCAL_COMMIT_INDEX_COLUMN, pa.int32()),
        pa.field(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, pa.bool_()),
        pa.field(HAS_RENAME_AMBIGUITY_COLUMN, pa.bool_()),
        pa.field(TOKEN_CHANGE_MASK_PRE_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHANGE_MASK_POST_COLUMN, pa.list_(pa.uint8())),
        pa.field(HUNK_ID_PER_TOKEN_COLUMN, pa.list_(pa.int32())),
        pa.field(EDIT_OP_PER_TOKEN_COLUMN, pa.list_(pa.uint8())),
        pa.field(CHANGED_CHUNK_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(
            CHANGED_CHUNK_SPANS_COLUMN,
            pa.list_(pa.struct([pa.field("start", pa.uint32()), pa.field("end", pa.uint32())])),
        ),
    ]
)


def open_jsonl(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


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


def _normalize_edges(raw_edges: list[dict] | None) -> list[dict[str, int]]:
    normalized: list[dict[str, int]] = []
    for edge in raw_edges or []:
        normalized.append({"from": int(edge.get("from", 0)), "to": int(edge.get("to", 0))})
    return normalized


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


def parse_record(raw: dict) -> dict:
    return {
        "text": raw.get("text", ""),
        "actual_token_count": _optional_int(raw.get("actual_token_count")),
        "structure_ids": raw.get("structure_ids", []),
        "chunk_boundaries": _normalize_chunk_boundaries(raw.get("chunk_boundaries")),
        "call_edges": _normalize_edges(raw.get("call_edges")),
        "type_edges": _normalize_edges(raw.get("type_edges")),
        "ast_depth": raw.get("ast_depth", []),
        "sibling_index": raw.get("sibling_index", []),
        "ast_node_type": raw.get("ast_node_type", []),
        "platform_info": json.dumps(raw["platform_info"]) if raw.get("platform_info") else None,
        "language_info": json.dumps(raw["language_info"]) if raw.get("language_info") else None,
        "build_info": json.dumps(raw["build_info"]) if raw.get("build_info") else None,
        "constituent_provenance": _decode_constituent_provenance(raw),
        "constituent_provenance_json": raw.get("constituent_provenance_json"),
        REPO_COLUMN: raw.get(REPO_COLUMN),
        FILEPATH_COLUMN: raw.get(FILEPATH_COLUMN),
        COMMIT_HASH_COLUMN: raw.get(COMMIT_HASH_COLUMN),
        TIMESTAMP_COLUMN: raw.get(TIMESTAMP_COLUMN),
        PARENT_HASHES_COLUMN: [str(item) for item in raw.get(PARENT_HASHES_COLUMN, [])],
        PARENT_COUNT_COLUMN: _optional_int(raw.get(PARENT_COUNT_COLUMN)),
        IS_MERGE_COMMIT_COLUMN: bool(raw.get(IS_MERGE_COMMIT_COLUMN)) if raw.get(IS_MERGE_COMMIT_COLUMN) is not None else None,
        AUTHOR_TIMESTAMP_COLUMN: raw.get(AUTHOR_TIMESTAMP_COLUMN),
        COMMIT_TIMESTAMP_COLUMN: raw.get(COMMIT_TIMESTAMP_COLUMN),
        REPO_STABLE_ID_COLUMN: raw.get(REPO_STABLE_ID_COLUMN),
        FILEPATH_STABLE_ID_COLUMN: raw.get(FILEPATH_STABLE_ID_COLUMN),
        FILE_LOCAL_COMMIT_INDEX_COLUMN: _optional_int(raw.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)),
        HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN: bool(raw.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, False)),
        HAS_RENAME_AMBIGUITY_COLUMN: bool(raw.get(HAS_RENAME_AMBIGUITY_COLUMN, False)),
        TOKEN_CHANGE_MASK_PRE_COLUMN: raw.get(TOKEN_CHANGE_MASK_PRE_COLUMN, []),
        TOKEN_CHANGE_MASK_POST_COLUMN: raw.get(TOKEN_CHANGE_MASK_POST_COLUMN, []),
        HUNK_ID_PER_TOKEN_COLUMN: raw.get(HUNK_ID_PER_TOKEN_COLUMN, []),
        EDIT_OP_PER_TOKEN_COLUMN: raw.get(EDIT_OP_PER_TOKEN_COLUMN, []),
        CHANGED_CHUNK_IDS_COLUMN: raw.get(CHANGED_CHUNK_IDS_COLUMN, []),
        CHANGED_CHUNK_SPANS_COLUMN: raw.get(CHANGED_CHUNK_SPANS_COLUMN, []),
    }


def flush_batch(writer: pq.ParquetWriter, batch: list[dict]) -> None:
    if not batch:
        return
    columns = {name: [record.get(name) for record in batch] for name in SCHEMA.names}
    writer.write_table(pa.table(columns, schema=SCHEMA))


def convert(input_path: str, output_path: str, row_group_size: int) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    batch: list[dict] = []
    writer = pq.ParquetWriter(output_path, SCHEMA)
    try:
        with open_jsonl(input_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                batch.append(parse_record(json.loads(line)))
                if len(batch) >= row_group_size:
                    flush_batch(writer, batch)
                    batch.clear()
        flush_batch(writer, batch)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-group-size", type=int, default=1000)
    args = parser.parse_args()
    convert(args.input, args.output, args.row_group_size)


if __name__ == "__main__":
    main()
