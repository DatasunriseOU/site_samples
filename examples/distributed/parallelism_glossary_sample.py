"""Public-safe glossary for the main parallel axes used by the MegaCpp POC code."""

GLOSSARY = {
    "pp": {
        "name": "pipeline parallelism",
        "ownership": "partitions model layers across pipeline stages so different stages can work on different micro-batches",
        "notes": "stage balance and overlap still depend on the schedule and microbatch count",
    },
    "dp": {
        "name": "data parallel sharding",
        "ownership": "shards parameters and optimizer state within each stage instead of fully replicating them everywhere",
        "notes": "this sample follows the MegaCpp POC's sharded FSDP-style interpretation, not classic full-replica DP",
    },
    "tp": {
        "name": "tensor parallelism",
        "ownership": "shards compatible weight dimensions across ranks",
        "notes": "head and tensor dimensions must divide cleanly across the TP degree",
    },
    "ep": {
        "name": "expert parallelism",
        "ownership": "partitions expert weights across expert peers and dispatches tokens between them",
        "notes": "EP requires an MoE layer layout and does not by itself solve routing balance or capacity planning",
    },
}

APPLY_ORDER = ("tp", "pp", "dp", "ep")


def describe_parallelism(mode: str) -> dict[str, str]:
    key = mode.lower()
    if key not in GLOSSARY:
        raise KeyError(f"unknown mode: {mode}")
    entry = GLOSSARY[key]
    return {
        "mode": key,
        "name": entry["name"],
        "ownership": entry["ownership"],
        "notes": entry["notes"],
    }


def describe_application_order() -> tuple[str, ...]:
    """Return the MegaCpp POC-grounded high-level application order for the axes."""

    return APPLY_ORDER
