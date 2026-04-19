"""Exact-token sparse telemetry sample.

What it is: a public-safe receipt for the MegaCpp POC exact-token sparse path
that records requested backend, actual backend, runtime mode, and reroute
reasons.

Why it exists: exact-token sparse attention can reroute to full attention or
stay on its sparse lane depending on what the indexer and backend actually
return.

What problem it solves: it keeps sparse runtime receipts honest by making the
semantic path and backend choice visible instead of silent.
"""

from __future__ import annotations


def build_exact_token_sparse_telemetry(
    *,
    indexer_type: str,
    requested_backend: str,
    routing_result_is_none: bool,
    seq_len: int,
) -> dict[str, object]:
    if routing_result_is_none:
        return {
            "path": "reroute_to_full",
            "reason": "indexer_returned_none",
            "indexer": indexer_type,
            "seq_len": seq_len,
        }

    return {
        "path": "exact_token",
        "indexer": indexer_type,
        "backend": requested_backend,
        "seq_len": seq_len,
        "requested_backend": requested_backend,
        "actual_backend": requested_backend,
        "runtime_mode": "exact_token",
    }
