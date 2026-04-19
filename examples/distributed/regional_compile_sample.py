"""Compile transformer blocks one by one before outer sharding wrappers.

This example shows the donor's regional-compile pattern in a public-safe form.
It exists because whole-model compile can hide submodules from later sharding
passes, while per-block compile keeps the compile boundary small and easier to
skip when a block family is known to be unstable.
"""

from __future__ import annotations

from collections.abc import Iterable


def iter_native_mamba3_children(module) -> Iterable[object]:
    """Yield children that look like upstream native Mamba3 modules.

    The donor skips these because upstream Triton TMA descriptors are not safe
    to push through the same static compile analysis path.
    """

    if bool(getattr(module, "_mamba3_native", False) or getattr(module, "_m2rnn_megatron_style", False)):
        yield module
        return
    for child in module.modules():
        child_module = type(child).__module__ or ""
        if child_module.startswith("mamba_ssm.modules.mamba3"):
            yield child


def apply_regional_compile_public(model, compile_kwargs: dict, *, tp_active: bool = False) -> dict[str, int]:
    """Compile each block in-place when the donor rules say that is safe.

    This mirrors the donor contract:
    1. run tensor/sequence parallel sharding first,
    2. compile block-by-block,
    3. let outer wrappers attach afterwards.
    """

    blocks = model.transformer.h
    compiled = 0
    skipped = 0

    if tp_active:
        return {"compiled": 0, "skipped": len(blocks)}

    for index, block in enumerate(blocks):
        inner = getattr(block, "block", block)
        if any(iter_native_mamba3_children(inner)):
            skipped += 1
            continue

        object.__setattr__(inner, "_skip_manual_checkpoint", True)
        try:
            block.compile(**compile_kwargs)
        except (AttributeError, TypeError):
            import torch

            blocks[index] = torch.compile(block, **compile_kwargs)
        compiled += 1

    return {"compiled": compiled, "skipped": skipped}
