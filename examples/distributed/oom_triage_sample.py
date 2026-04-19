"""Public OOM triage helper for article references."""


def estimate_memory_budget(params_gb: float, activations_gb: float, optimizer_gb: float) -> float:
    return round(params_gb + activations_gb + optimizer_gb, 2)


def dominant_pressure(params_gb: float, activations_gb: float, optimizer_gb: float) -> str:
    buckets = {
        "parameters": params_gb,
        "activations": activations_gb,
        "optimizer": optimizer_gb,
    }
    return max(buckets, key=buckets.get)
