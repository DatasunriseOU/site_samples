"""Adapter and LoRA runtime surface sample.

What it is: a public-safe summary of the runtime adapter metadata contract used
for hybrid LoRA stacks.

Why it exists: the MegaCpp POC supports several adapter flavors, and runtime
composition only works when the adapter snapshot preserves enough metadata.

What problem it solves: it shows which adapter traits are carried forward so
hot-swapping or composing adapters does not guess ranks, quantization, or
specialized LoRA variants.
"""

from __future__ import annotations

from collections.abc import Mapping


def summarize_runtime_adapter_meta(root_metadata: Mapping[str, object]) -> dict[str, object]:
    qlora_meta = root_metadata.get("qlora_quant_metadata")
    dylora_meta = root_metadata.get("dylora_runtime_metadata")
    rank = root_metadata.get("rank")
    alpha = root_metadata.get("alpha")

    return {
        "rank": rank,
        "alpha": alpha,
        "use_dora": bool(root_metadata.get("use_dora", False)),
        "use_vera": bool(root_metadata.get("use_vera", False)),
        "use_qlora": qlora_meta is not None or bool(root_metadata.get("use_qlora", False)),
        "use_dylora": dylora_meta is not None or bool(root_metadata.get("use_dylora", False)),
        "qlora_block_size": root_metadata.get("qlora_block_size"),
        "qlora_compute_dtype": root_metadata.get("qlora_compute_dtype"),
        "dylora_current_rank": root_metadata.get("dylora_current_rank"),
        "why_it_matters": [
            "composition has to know whether the adapter is plain LoRA, DoRA, VeRA, QLoRA, or DyLoRA",
            "quantized adapters need block-size and compute-dtype metadata",
            "dynamic-rank adapters need the active runtime rank to avoid mismatched merges",
        ],
    }


def adapter_stack_receipt() -> dict[str, object]:
    return {
        "supported_runtime_families": ["lora", "dora", "vera", "qlora", "dylora"],
        "composition_rule": "compose LoRA deltas in delta space; reject mathematically unsafe adapter mixes",
        "online_use_case": "keep adapter snapshots self-describing so they can stay unmerged during continued adaptation",
    }
