"""Block-scoped CUDA graph validation for compiled model regions.

What it is: a donor-based excerpt of the runtime check that validates requested
CUDA-graph block types and forces the matching Inductor env vars on.

Why it exists: explicit graph requests are easy to misconfigure when the model
changes and the requested block names no longer exist.

What problem it solves: it turns a silent "graph flags were passed but nothing
was captured" failure mode into a visible summary of found and missing block
types, while preserving the donor runtime behavior of enabling the graph env
vars when explicit graph blocks are requested.
"""

from __future__ import annotations

import os
from collections.abc import Iterable


def validate_cuda_graph_blocks(
    *,
    named_modules: Iterable[tuple[str, object]],
    requested_blocks: str,
    device_type: str,
) -> dict[str, object]:
    """Return a public-safe summary of requested CUDA-graph block coverage."""
    inductor_active = (
        os.environ.get("TORCHINDUCTOR_TRITON_CUDAGRAPHS", "0") == "1"
        and os.environ.get("TORCH_COMPILE_CUDAGRAPH_TREES", "0") == "1"
    )
    summary: dict[str, object] = {
        "inductor_active": inductor_active,
        "requested": [],
        "found": {},
        "missing": [],
        "message": None,
    }

    if device_type != "cuda":
        summary["message"] = "CUDA graph validation is only relevant on CUDA devices."
        return summary

    requested = [item.strip() for item in requested_blocks.split(",") if item.strip()]
    summary["requested"] = requested
    if not requested:
        if inductor_active:
            summary["message"] = (
                "Inductor CUDA graph trees are active via env defaults; compiled "
                "blocks can still be captured automatically."
            )
        else:
            summary["message"] = "No explicit CUDA-graph block types were requested."
        return summary

    found: dict[str, int] = {}
    for _name, module in named_modules:
        module_type = type(module).__name__
        if module_type in requested:
            found[module_type] = found.get(module_type, 0) + 1

    missing = [module_type for module_type in requested if module_type not in found]
    if found and not inductor_active:
        # Donor behavior: explicit block capture should force the Inductor graph
        # env vars so the request cannot silently degrade into a no-op.
        os.environ["TORCHINDUCTOR_TRITON_CUDAGRAPHS"] = "1"
        os.environ["TORCH_COMPILE_CUDAGRAPH_TREES"] = "1"
        inductor_active = True

    summary["inductor_active"] = inductor_active
    summary["found"] = found
    summary["missing"] = missing
    if found:
        formatted = ", ".join(f"{name}={count}" for name, count in sorted(found.items()))
        summary["message"] = f"Inductor CUDA graph trees active for compiled blocks: {formatted}"
    elif missing:
        summary["message"] = f"Requested CUDA-graph block types not found: {missing}"
    return summary
