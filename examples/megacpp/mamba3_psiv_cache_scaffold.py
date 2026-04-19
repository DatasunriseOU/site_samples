"""Near-copy MegaCpp POC example: Mamba3 PsiV cache scaffold.

This sample stays honest about the current public contract: the cache is an
explicit scaffold, not a silently enabled optimization. If the gate is turned
on while the implementation is still absent, the correct behavior is to refuse
the run rather than hide the missing feature behind a fallback.
"""

from __future__ import annotations


def is_enabled(env: dict[str, str]) -> bool:
    return env.get("CPPMEGA_MAMBA3_P2_PSIV_CACHE", "0") == "1"


def refuse_if_gated(env: dict[str, str]) -> str | None:
    if is_enabled(env):
        return (
            "CPPMEGA_MAMBA3_P2_PSIV_CACHE=1 but the cache is a scaffold only; "
            "unset the variable to run the baseline kernel path"
        )
    return None


def scaffold_status() -> dict[str, object]:
    return {
        "phase_a_python_precompute": "not implemented",
        "phase_b_gmem_pool": "not implemented",
        "phase_c_bwd_fwd_cache_input": "not implemented",
        "phase_c_bwd_bwd_cache_input": "not implemented",
        "failure_mode": "fail closed when explicitly gated on",
    }
