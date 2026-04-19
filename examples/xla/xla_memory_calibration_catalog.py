"""XLA startup calibration catalog sample.

This example shows how the TPU startup path records which launch candidates
worked and which ones failed. The point is to stop retrying the exact same bad
parallelism and memory configuration after a compile-time OOM.
"""

from __future__ import annotations


def build_candidate_signature(signature_base: dict, candidate: dict) -> dict:
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


def make_record(signature_hash: str, *, outcome: str, failure_type: str | None = None) -> dict:
    startup = {"outcome": outcome}
    if failure_type is not None:
        startup["failure_type"] = failure_type
    return {"signature_hash": signature_hash, "startup": startup}


def rerank_candidates(candidates: list[dict], *, signature_base: dict, records: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for retry_rank, candidate in enumerate(candidates):
        signature = build_candidate_signature(signature_base, candidate)
        parallelism = signature["parallelism"]
        signature_hash = (
            f"tp{parallelism['tp_degree']}-ep{parallelism['ep_degree']}"
            f"-dp{parallelism['dp_degree']}-bs{parallelism['device_batch_size']}"
            f"-ga{parallelism['grad_accum_steps']}"
        )
        related = [r for r in records if r.get("signature_hash") == signature_hash]
        success_count = sum(r.get("startup", {}).get("outcome") == "success" for r in related)
        compile_failure_count = sum(
            r.get("startup", {}).get("failure_type") in {"CompileTimeHbmOom", "XlaResourceExhausted"}
            for r in related
        )
        enriched.append(
            {
                **candidate,
                "retry_rank": retry_rank,
                "signature_hash": signature_hash,
                "calibration": {
                    "success_count": success_count,
                    "compile_failure_count": compile_failure_count,
                },
                "calibration_hit": bool(related),
            }
        )

    return sorted(
        enriched,
        key=lambda item: (
            -item["calibration"]["success_count"],
            item["calibration"]["compile_failure_count"],
            item["retry_rank"],
        ),
    )
