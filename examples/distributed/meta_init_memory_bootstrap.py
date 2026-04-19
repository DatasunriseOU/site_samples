"""Public-safe meta-init bootstrap example for large-model bringup."""


def bootstrap_summary() -> list[str]:
    return [
        "Construct modules on empty or meta storage first.",
        "Materialize only the shard-local tensors on the target device.",
        "Use this to avoid full-model CPU residency before sharding.",
    ]
