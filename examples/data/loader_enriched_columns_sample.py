"""Loader-side enriched column normalization excerpt.

This example shows how the MegaCpp POC loader reads optional enriched parquet
columns without crashing the whole batch on malformed JSON. The problem it
solves is robustness: enrichment is useful only if older shards and partial
metadata stay readable.
"""

from __future__ import annotations

import json
from typing import Any


def warn_bad_json_value(column_name: str, row_idx: int, value: Any, exc: Exception) -> None:
    preview = repr(value)
    if len(preview) > 120:
        preview = preview[:117] + "..."
    print(
        f"[WARN] Malformed enriched JSON in {column_name} row {row_idx}: {exc}; "
        f"falling back to default metadata ({preview})"
    )


def decode_optional_json_value(
    value: Any,
    *,
    column_name: str,
    row_idx: int,
    default_value: Any,
    expected_type: type | tuple[type, ...] | None,
) -> Any:
    parsed = value
    if value is None or value == "":
        return default_value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            warn_bad_json_value(column_name, row_idx, value, exc)
            return default_value
    if parsed is None:
        return default_value
    if expected_type is not None and not isinstance(parsed, expected_type):
        if isinstance(expected_type, tuple):
            expected_name = "/".join(item.__name__ for item in expected_type)
        else:
            expected_name = expected_type.__name__
        warn_bad_json_value(
            column_name,
            row_idx,
            value,
            TypeError(f"expected {expected_name}, got {type(parsed).__name__}"),
        )
        return default_value
    return parsed


def decode_optional_json_column(raw_values: list[Any], *, column_name: str) -> list[list[Any]]:
    decoded: list[list[Any]] = []
    for row_idx, value in enumerate(raw_values):
        decoded.append(
            decode_optional_json_value(
                value,
                column_name=column_name,
                row_idx=row_idx,
                default_value=[],
                expected_type=list,
            )
        )
    return decoded


def read_enriched_json_columns(row_group: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Decode the JSON-backed enriched columns used by the loader fallback path."""
    decoded: dict[str, list[Any]] = {}
    for column_name in ("chunk_boundaries", "call_edges", "type_edges"):
        if column_name in row_group:
            decoded[column_name] = decode_optional_json_column(
                row_group[column_name],
                column_name=column_name,
            )
    return decoded
