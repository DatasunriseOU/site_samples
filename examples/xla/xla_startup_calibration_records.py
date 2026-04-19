"""TPU startup calibration record sample.

This shows the record shape used to keep successful and failed startup attempts
from the first compile window. It exists so later launches can prefer settings
that already worked and avoid repeating compile-time HBM failures.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_signature_hash(signature: Mapping[str, object]) -> str:
    payload = repr(sorted(signature.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def make_startup_record(
    *,
    signature: Mapping[str, object],
    estimator: Mapping[str, object],
    startup: Mapping[str, object],
    observed: Mapping[str, object] | None = None,
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "signature": dict(signature),
        "signature_hash": build_signature_hash(signature),
        "estimator": dict(estimator),
        "startup": dict(startup),
        "observed": dict(observed or {}),
        "source": dict(source or {}),
    }
    est_total = record["estimator"].get("estimated_with_runtime_gb") if isinstance(record["estimator"], dict) else None
    post_step0 = record["observed"].get("post_step0") if isinstance(record["observed"], dict) else None
    peak = post_step0.get("peak_gb") if isinstance(post_step0, dict) else None
    if isinstance(est_total, (int, float)) and isinstance(peak, (int, float)):
        record["calibration"] = {
            "delta_post_step0_peak_gb": round(float(peak) - float(est_total), 2)
        }
    return record
