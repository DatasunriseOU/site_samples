"""Mark only the batch dimension as dynamic before compiled block calls.

This example shows the donor helper used to avoid recompiling every time the
 device batch size changes. It keeps sequence length concrete, because some
 compile grids break when the time dimension becomes symbolic.
"""

from __future__ import annotations

import torch


def install_batch_dynamic_hooks(model):
    """Add temporary forward pre-hooks to compiled blocks.

    The donor uses this because compiled block inputs are hidden states, not the
    original token tensor, so symbolic batch information does not automatically
    flow into every regional-compile boundary.
    """

    handles = []

    def _hook(module, args):
        for value in args:
            if torch.is_tensor(value) and value.ndim >= 1:
                if value.size(0) <= 1:
                    torch._dynamo.maybe_mark_dynamic(value, 0)
                else:
                    torch._dynamo.mark_dynamic(value, 0)
        return args

    for block in getattr(model.transformer, "h", []):
        if hasattr(block, "_orig_mod"):
            handles.append(block.register_forward_pre_hook(_hook))

    return handles


def remove_batch_dynamic_hooks(handles) -> None:
    """Tear down the temporary hooks after warmup finishes."""

    for handle in handles:
        handle.remove()
