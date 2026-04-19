"""Public-safe schedule-phase residency helper."""


def residency_schedule(stages: int, microbatches: int) -> dict[str, int]:
    return {
        "warmup": min(max(stages - 1, 1), microbatches),
        "steady_state": min(stages, microbatches),
        "drain": min(max(stages - 1, 1), microbatches),
    }
