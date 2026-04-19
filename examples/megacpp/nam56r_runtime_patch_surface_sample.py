"""NAM56R runtime patch-surface sample.

What it is: a public-safe summary of the extra runtime surfaces that sit on top
of the plain recipe for NAM56R.

Why it exists: some parts of the model run through direct recipe flags, while
others rely on schedule, fused-loss, or shared-spec patches around Megatron.

What problem it solves: it separates "defined by the recipe" from "enabled by a
runtime patch surface" so public examples do not imply everything is native to
one upstream config object.
"""

from __future__ import annotations


def build_runtime_patch_surface_receipt() -> dict[str, object]:
    return {
        "recipe_native": [
            "hybrid pattern expansion",
            "MoE size and top-k",
            "MLA dimensions",
            "parallelism defaults",
        ],
        "runtime_patch_surfaces": {
            "mtp_native_hopper_ce": {
                "purpose": "route main head and MTP depths through one native linear-CE fusion path",
                "scope": "training loss path",
            },
            "hybrid_schedule_plan": {
                "purpose": "keep hybrid Mamba, DSA, MoE, and MTP scheduling rules consistent",
                "scope": "pipeline/post-process runtime plan",
            },
            "mla_shared_spec": {
                "purpose": "share MLA layer-spec helpers and memory-saving choices across selective attention layers",
                "scope": "attention-layer construction",
            },
        },
        "notes": [
            "The recipe defines the model shape, but some live behaviors still depend on explicit runtime patch points.",
            "This is why public launcher docs should mention both recipe values and patch surfaces.",
        ],
    }
