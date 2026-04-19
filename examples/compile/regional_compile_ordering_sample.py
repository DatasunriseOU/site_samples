"""Regional-compile ordering contract.

What it is: a public-safe ordering receipt for the MegaCpp POC regional-compile
path in distributed training.

Why it exists: applying compile in the wrong order breaks DTensor tracing or
pushes FSDP2 into a backward path that cannot see the intended compiled leaves.

What problem it solves: it records the exact ordering that keeps TP and SP
visible to compile while still letting FSDP2 wrap the right modules afterward.
"""

from __future__ import annotations


def regional_compile_ordering_receipt() -> dict[str, object]:
    ordered_steps = [
        "build_model",
        "apply_tensor_parallel_and_sequence_parallel",
        "apply_regional_compile_to_supported_blocks",
        "wrap_with_fsdp2_or_other_outer_runtime",
    ]
    return {
        "ordered_steps": ordered_steps,
        "must_hold": [
            "TP+SP must happen before regional compile so torch.compile traces the DTensor-facing path",
            "regional compile must happen before FSDP2 so the outer wrapper sees already-compiled leaves",
        ],
        "block_exclusions": {
            "eblock": "MoE EBlocks are skipped from regional compile in the MegaCpp POC lane",
        },
    }
