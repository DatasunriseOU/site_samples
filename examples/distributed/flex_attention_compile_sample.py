"""FlexAttention compile sample.

This example shows how the GPU attention path loads FlexAttention, compiles it
only on CUDA, and keeps the optional FLASH backend probe separate. It exists
because importing or compiling the wrong surface drags extra compiler and graph
machinery into cold paths. The problem it solves is keeping the hot CUDA path
fast without making CPU tests or unsupported CUDA envs fail opaquely.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Callable

import torch


def load_flex_attention() -> tuple[Callable | None, Callable | None]:
    """Load the donor FlexAttention surface with CUDA-only compilation.

    Grounded donor rule: CPU import and test paths must not drag Inductor or
    cudagraph machinery in just to prove that FlexAttention imports. The donor
    therefore wraps `flex_attention` with `torch.compile(...,
    mode="max-autotune-no-cudagraphs")` only when CUDA is available.
    """

    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError:
        return None, None
    compiled = (
        torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
        if torch.cuda.is_available()
        else flex_attention
    )
    return compiled, flex_attention


@lru_cache(maxsize=1)
def probe_flex_flash_backend() -> tuple[bool, str | None]:
    """Probe the optional FLASH/CuTe backend exactly like the donor surface.

    The donor does this upfront so unsupported systems fail with a clear reason
    instead of surfacing an opaque import error from generated code deep inside
    the compiled attention path.
    """

    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - mirrors donor failure path
        return False, f"CUDA capability probe failed: {type(exc).__name__}: {exc}"
    if major not in (9, 10, 11):
        return False, f"unsupported compute capability sm{major}{minor}"
    for module_name in ("cutlass", "cuda.bindings.driver", "flash_attn.cute.interface"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - mirrors donor failure path
            return False, f"{module_name} import failed: {type(exc).__name__}: {exc}"
    return True, None


def compile_safe_softcap_note() -> tuple[str, ...]:
    """Return the donor's compile-safety note for score modifiers."""

    return (
        "compiled CUDA paths should use compile-safe score modifiers",
        "cached wrapper helpers can become graph breaks under Dynamo",
        "cold eager helpers are still fine for CPU tests and unit-scale callers",
    )
