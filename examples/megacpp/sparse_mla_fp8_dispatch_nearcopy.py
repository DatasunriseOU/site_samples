"""Near-copy MegaCpp POC example: SparseMLA FP8 dispatch contract.

This sample keeps the central failure surface close to the real reproducer:
Transformer Engine Float8Tensor wrappers are not ordinary bf16 tensors even
when their logical dtype looks bf16 to generic dispatch code.

The public point is simple:
- raw dispatch can see a wrapper with a NULL-facing `data_ptr()` and hidden FP8 storage
- a dequantize fallback restores correctness at extra bandwidth cost
- an FP8-aware dispatch path keeps the contract explicit and avoids silent downgrade
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Float8TensorSurface:
    logical_dtype: str = "bfloat16"
    storage_dtype: str = "float8_e4m3"
    public_data_ptr: int = 0
    storage_data_ptr: int = 4096

    def contiguous(self) -> "Float8TensorSurface":
        return self

    def to_bfloat16(self) -> "Float8TensorSurface":
        return self

    def dequantize(self) -> dict[str, str | int]:
        return {
            "dtype": "bfloat16",
            "data_ptr": self.storage_data_ptr,
            "origin": "dequantized",
        }


def raw_dispatch_surface(tensor: Float8TensorSurface) -> dict[str, object]:
    return {
        "logical_dtype": tensor.logical_dtype,
        "storage_dtype": tensor.storage_dtype,
        "public_data_ptr": tensor.public_data_ptr,
        "storage_data_ptr": tensor.storage_data_ptr,
        "kernel_choice": "bf16" if tensor.logical_dtype == "bfloat16" else "fp8",
        "hazard": tensor.public_data_ptr == 0,
    }


def dequantize_fallback_surface(tensor: Float8TensorSurface) -> dict[str, object]:
    dense = tensor.dequantize()
    return {
        "kernel_choice": "bf16",
        "input_dtype": dense["dtype"],
        "data_ptr": dense["data_ptr"],
        "bandwidth_tradeoff": "extra dequantize / requantize path",
    }


def fp8_dispatch_surface(tensor: Float8TensorSurface) -> dict[str, object]:
    return {
        "kernel_choice": "fp8",
        "input_dtype": tensor.storage_dtype,
        "data_ptr": tensor.storage_data_ptr,
        "bandwidth_tradeoff": "no explicit dequantize round trip",
    }


def compare_dispatch_paths() -> dict[str, dict[str, object]]:
    tensor = Float8TensorSurface()
    return {
        "raw_dispatch": raw_dispatch_surface(tensor),
        "dequantize_fallback": dequantize_fallback_surface(tensor),
        "fp8_aware_dispatch": fp8_dispatch_surface(tensor),
    }
