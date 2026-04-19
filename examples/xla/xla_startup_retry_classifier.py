"""TPU startup retry classifier example.

This shows how TPU startup failures are sorted into retryable and non-retryable
classes. It exists so the launcher can fall back only on failures that were
actually seen during early XLA compile bring-up, instead of looping forever on
the same bad graph.
"""

from __future__ import annotations


def classify_xla_startup_failure(exc: BaseException) -> dict[str, object] | None:
    """Classify the retryable TPU startup failures seen in startup receipts."""
    text = f"{type(exc).__name__}: {exc}"
    lower = text.lower()
    if ("compile permanent error" in lower and "memory space hbm" in lower) or (
        "resource_exhausted" in lower and "hbm" in lower and "compile" in lower
    ):
        return {
            "failure_type": "CompileTimeHbmOom",
            "retryable": True,
            "message": text,
        }
    if "program allocation failure" in lower:
        return {
            "failure_type": "ProgramAllocationFailure",
            "retryable": True,
            "message": text,
        }
    if "buffer allocation failure" in lower:
        return {
            "failure_type": "BufferAllocationFailure",
            "retryable": True,
            "message": text,
        }
    if "resource exhausted" in lower and "hbm" in lower:
        return {
            "failure_type": "XlaResourceExhausted",
            "retryable": True,
            "message": text,
        }
    return None


def is_retryable_xla_startup_window(*, device_type: str, step: int) -> bool:
    """Allow fallback only during startup and the first post-step0 compile window."""
    return device_type == "xla" and step <= 1
