"""Public excerpt.

Source: MegaCpp tensor-parallel sizing excerpt
Purpose: show explicit TP partition sizing instead of hand-wavy sharding claims
Edited for clarity.
"""

def compute_mamba3_tp_partition_sizes(total_sizes, tp_size):
    if any(size % tp_size != 0 for size in total_sizes):
        raise ValueError("partition sizes must divide evenly across tensor-parallel ranks")
    return [size // tp_size for size in total_sizes]
