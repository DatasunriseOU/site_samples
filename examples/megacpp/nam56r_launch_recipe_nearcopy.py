"""Near-copy MegaCpp POC example: NAM56R launch-policy contract.

This sample keeps the public split visible: generated Megatron arguments are
one surface, but launch policy is another. A real NAM56R launcher also carries
environment, parallelism, and runtime safety flags that are not implied by the
translated layer pattern alone.
"""

from __future__ import annotations


def build_launch_policy() -> dict[str, object]:
    return {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "mtp_depths": 1,
        "force_author_mamba3": True,
        "requires_custom_r_seam": True,
    }


def build_runtime_env() -> dict[str, str]:
    return {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
    }


def build_launch_contract() -> dict[str, object]:
    return {
        "translated_pattern": "*EME*EME*EMR/*-",
        "launch_policy": build_launch_policy(),
        "runtime_env": build_runtime_env(),
        "notes": "generated args and fixed runtime policy are separate surfaces",
    }
