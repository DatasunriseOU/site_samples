"""Donor-based VPP warmup-count helper excerpt.

This is a public-safe excerpt of the donor's ``get_vpp_warmup_count`` helper.
It keeps the validated warmup/steady-state formula used for analysis and tests.
"""


def get_vpp_warmup_count(
    *,
    pp_rank: int,
    pp_degree: int,
    num_microbatches: int,
    num_model_chunks: int,
) -> dict[str, int]:
    """Return the VPP warmup/remaining-step counts for one pipeline rank."""

    if pp_degree < 1 or num_model_chunks < 1 or num_microbatches < 1:
        raise ValueError("pp_degree, num_model_chunks, and num_microbatches must be positive")
    if not 0 <= pp_rank < pp_degree:
        raise ValueError(f"pp_rank={pp_rank} out of range for pp_degree={pp_degree}")

    microbatch_group_size = pp_degree
    total = num_microbatches * num_model_chunks
    warmup = (pp_degree - pp_rank - 1) * 2
    warmup += (num_model_chunks - 1) * microbatch_group_size
    warmup = min(warmup, total)
    remaining = total - warmup

    return {
        "warmup_microsteps": warmup,
        "remaining_microsteps": remaining,
        "pp_rank": pp_rank,
        "pp_degree": pp_degree,
        "num_microbatches": num_microbatches,
        "num_model_chunks": num_model_chunks,
    }
