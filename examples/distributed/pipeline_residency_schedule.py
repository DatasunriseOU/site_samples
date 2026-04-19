"""MegaCpp POC-based VPP schedule-regime helper excerpt.

This public-safe helper keeps the MegaCpp POC's proven operating-regime checks for
interleaved pipeline schedules.
"""


def validate_vpp_microbatch_regime(
    *,
    pp_degree: int,
    num_model_chunks: int,
    num_microbatches: int,
) -> dict[str, int | bool]:
    """Validate the default VPP operating regime used by the MegaCpp POC helpers."""

    if pp_degree < 1 or num_model_chunks < 1 or num_microbatches < 1:
        raise ValueError("pp_degree, num_model_chunks, and num_microbatches must be positive")
    if num_microbatches % pp_degree != 0:
        raise ValueError("num_microbatches must be a multiple of pp_degree")

    microbatch_group_size = pp_degree
    total = num_microbatches * num_model_chunks
    warmup_rank0 = min(
        (pp_degree - 1) * 2 + (num_model_chunks - 1) * microbatch_group_size,
        total,
    )
    all_warmup = warmup_rank0 >= total
    if all_warmup:
        raise ValueError("all-warmup regime is invalid for VPP")
    return {
        "pp_degree": pp_degree,
        "num_model_chunks": num_model_chunks,
        "num_microbatches": num_microbatches,
        "rank0_warmup_microsteps": warmup_rank0,
        "all_warmup": all_warmup,
    }
