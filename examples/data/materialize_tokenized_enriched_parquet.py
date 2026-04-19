#!/usr/bin/env python3
"""Token-level enriched parquet materialization example.

This example shows how char-level enrichment is turned into token-aligned
training rows. The point is to preserve structure, AST, and dependency signals
after tokenization so the model can learn from code semantics instead of raw
text alone.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TOKENIZED_ENRICHED_COLUMNS = (
    "token_ids",
    "platform_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
)


def _decode_json_like(value):
    while isinstance(value, str) and value:
        decoded = json.loads(value)
        if decoded == value:
            break
        value = decoded
    return value


def _list_parquet_files(input_dir: str) -> list[Path]:
    return sorted(Path(input_dir).glob("*.parquet"), key=lambda path: path.name)


def _table_to_docs(table: pa.Table) -> list[dict]:
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    docs: list[dict] = []
    for row_idx in range(table.num_rows):
        row = {name: values[row_idx] for name, values in columns.items()}
        for field in ("chunk_boundaries", "call_edges", "type_edges"):
            value = row.get(field)
            if value is None:
                continue
            value = _decode_json_like(value)
            row[field] = [_decode_json_like(item) for item in value] if isinstance(value, list) else value
        for field in ("platform_info", "language_info", "build_info"):
            value = row.get(field)
            if value is not None:
                row[field] = _decode_json_like(value)
        docs.append(row)
    return docs


def _merge_table_with_tokenized(table: pa.Table, tokenized_rows: list[dict]) -> pa.Table:
    merged = {name: table.column(name) for name in table.column_names}
    for column_name in TOKENIZED_ENRICHED_COLUMNS:
        merged[column_name] = pa.array([row.get(column_name, []) for row in tokenized_rows])
    return pa.table(merged)


def _copy_metadata_files(input_dir: str, output_dir: str) -> None:
    for name in ("_COMPLETE",):
        src = Path(input_dir) / name
        if src.exists():
            Path(output_dir, name).write_text(src.read_text())


def materialize_tokenized_fields(docs: list[dict]) -> list[dict]:
    """Build token-aligned enriched columns from char-level document metadata."""

    rows: list[dict] = []
    for doc in docs:
        structure_ids = list(doc.get("structure_ids") or [])
        token_count = len(structure_ids)
        rows.append(
            {
                "token_ids": list(range(token_count)),
                "platform_ids": [0] * token_count,
                "token_structure_ids": structure_ids,
                "token_dep_levels": [0] * token_count,
                "token_ast_depth": [0] * token_count,
                "token_sibling_index": [0] * token_count,
                "token_ast_node_type": [0] * token_count,
            }
        )
    return rows


def rewrite_dataset(
    input_dir: str,
    output_dir: str,
    *,
    row_group_size: int,
    max_files: int,
    materializer: Callable[[list[dict]], list[dict]] = materialize_tokenized_fields,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = _list_parquet_files(input_dir)
    if max_files > 0:
        parquet_files = parquet_files[:max_files]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")

    for index, src_path in enumerate(parquet_files, start=1):
        dst_path = Path(output_dir) / src_path.name
        print(f"[{index}/{len(parquet_files)}] {src_path.name}")
        table = pq.read_table(src_path)
        docs = _table_to_docs(table)
        tokenized_rows = materializer(docs)
        merged = _merge_table_with_tokenized(table, tokenized_rows)
        pq.write_table(merged, dst_path, row_group_size=row_group_size, compression="snappy")

    _copy_metadata_files(input_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--row-group-size", type=int, default=1024)
    args = parser.parse_args()
    rewrite_dataset(
        args.input_dir,
        args.output_dir,
        row_group_size=args.row_group_size,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
