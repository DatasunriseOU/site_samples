"""TPU startup calibration catalog example.

This example shows the donor-backed record format used to remember which TPU
startup candidates succeeded or failed. It exists so repeated launches do not
retry the same bad sharding or batch-size combination after a compile-time or
startup OOM.

The problem it solves is wasted time: without a catalog, every fresh launch can
rediscover the same failure frontier.
"""

from __future__ import annotations

import hashlib
import json


def build_candidate_signature(signature_base: dict, candidate: dict) -> dict:
    """Public-safe excerpt of the donor launch signature contract."""

    return {
        "code": dict(signature_base.get("code", {})),
        "hardware": dict(signature_base.get("hardware", {})),
        "model": dict(signature_base.get("model", {})),
        "parallelism": {
            **dict(signature_base.get("parallelism", {})),
            "tp_degree": candidate.get("tp_degree", 1),
            "ep_degree": candidate.get("ep_degree", 1),
            "dp_degree": candidate.get("dp_degree", 1),
            "device_batch_size": candidate.get("device_batch_size", 1),
            "grad_accum_steps": candidate.get("grad_accum_steps", 1),
            "gradient_checkpointing": bool(candidate.get("gradient_checkpointing", False)),
            "fsdp": bool(candidate.get("fsdp", False)),
        },
        "features": dict(signature_base.get("features", {})),
    }


def build_signature_hash(signature: dict) -> str:
    """Hash the candidate contract the same way the donor catalog indexes entries."""

    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_record(signature_hash: str, *, outcome: str, failure_type: str | None = None) -> dict:
    startup = {"outcome": outcome}
    if failure_type is not None:
        startup["failure_type"] = failure_type
    return {"signature_hash": signature_hash, "startup": startup}


def rerank_candidates(candidates: list[dict], *, signature_base: dict, records: list[dict]) -> list[dict]:
    """Push known-bad launch shapes behind unseen or known-good candidates."""

    scored: list[tuple[tuple[int, int], dict]] = []
    known = {record["signature_hash"]: record for record in records}
    for retry_rank, candidate in enumerate(candidates):
        signature = build_candidate_signature(signature_base, candidate)
        signature_hash = build_signature_hash(signature)
        record = known.get(signature_hash)
        outcome = None if record is None else record.get("startup", {}).get("outcome")
        penalty = 0 if outcome in (None, "pass") else 1
        scored.append(((penalty, retry_rank), candidate))
    scored.sort(key=lambda item: item[0])
    return [candidate for _, candidate in scored]


def calibration_notes() -> tuple[str, ...]:
    return (
        "The signature includes code, hardware, model, parallelism, and feature state because any of them can move the startup memory frontier.",
        "A failed signature should not be retried first on the next run.",
        "The catalog is about startup viability, not final model quality.",
    )
