"""Dense FA4 execute-proof manifest builder.

What it is: the rollout-side helper that emits a runnable CUDA smoke command for
the dense/full FA4 path.

Why it exists: import success is not enough for a backend migration. The MegaCpp POC
helper materializes a concrete forward pass and records whether the output shape
and NaN checks pass.

What problem it solves: it separates rollout orchestration from runtime claims,
so preview-only manifests cannot be mistaken for promotion-grade receipts.
"""

from __future__ import annotations

from typing import Any


def torch_dtype_expr(dtype_name: str) -> str:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return "torch.bfloat16"
    if normalized in {"fp16", "float16", "half"}:
        return "torch.float16"
    raise ValueError(f"unsupported dense FA4 execute-proof dtype: {dtype_name!r}")


def build_dense_fa4_smoke_python(*, shape: dict[str, Any], report_json: str, environment: str) -> str:
    """Minimal MegaCpp POC-based version of `_build_dense_fa4_smoke_python`.

    Grounding:
    - MegaCpp dense FA4 execute-proof helper surface
    - backend import: `flash_attn.cute.interface.flash_attn_func`

    The key operational change is that the emitted script checks a real CUDA
    forward pass and writes a machine-readable payload. That keeps backend
    rollout discussion tied to execution evidence instead of import-only checks.
    """
    batch_size = int(shape["batch_size"])
    seq_len = int(shape["seq_len"])
    n_head = int(shape["n_head"])
    n_kv_head = int(shape["n_kv_head"])
    head_dim = int(shape["head_dim"])
    dtype_expr = torch_dtype_expr(str(shape["dtype"]))
    q_shape = (batch_size, seq_len, n_head, head_dim)
    kv_shape = (batch_size, seq_len, n_kv_head, head_dim)
    expected_shape = [batch_size, seq_len, n_head, head_dim]

    parts = [
        "import json",
        "from pathlib import Path",
        "import torch",
        "from flash_attn.cute.interface import flash_attn_func as _fa4_func",
        f"_report_path = Path({report_json!r})",
        "if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for dense/full FA4 execute-proof smoke')",
        f"q = torch.randn({q_shape!r}, device='cuda', dtype={dtype_expr})",
        f"k = torch.randn({kv_shape!r}, device='cuda', dtype={dtype_expr})",
        f"v = torch.randn({kv_shape!r}, device='cuda', dtype={dtype_expr})",
        "out = _fa4_func(q, k, v, causal=True)",
        "torch.cuda.synchronize()",
        "is_tuple = isinstance(out, tuple)",
        "head = out[0] if is_tuple else out",
        "output_shape = list(head.shape)",
        f"output_shape_valid = output_shape == {expected_shape!r}",
        "output_no_nan = not bool(torch.isnan(head).any().item())",
        "device_index = torch.cuda.current_device()",
        "payload = {"
        f"'environment': {environment!r}, "
        "'import_success': True, "
        "'forward_success': True, "
        "'output_shape_valid': output_shape_valid, "
        "'output_no_nan': output_no_nan, "
        "'execute_proof_passed': bool(output_shape_valid and output_no_nan), "
        "'output_shape': output_shape, "
        "'output_dtype': str(head.dtype), "
        "'device_name': torch.cuda.get_device_name(device_index)"
        "}",
        "_report_path.parent.mkdir(parents=True, exist_ok=True)",
        "_report_path.write_text(json.dumps(payload, sort_keys=True) + '\\n', encoding='utf-8')",
        "print(json.dumps(payload, sort_keys=True))",
    ]
    return "\n".join(parts)
