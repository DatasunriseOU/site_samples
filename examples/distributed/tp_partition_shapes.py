"""Public-safe tensor-parallel partition-size sketch."""


def partition_size(hidden_size: int, tp_size: int) -> int:
    if hidden_size % tp_size != 0:
        raise ValueError("hidden_size must divide evenly across tensor-parallel ranks")
    return hidden_size // tp_size
