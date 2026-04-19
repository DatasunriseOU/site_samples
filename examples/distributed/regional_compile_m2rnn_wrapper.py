"""Regional-compile-safe M2RNN wrapper for Megatron blocks.

What it is: the adapter that lets a custom recurrent mixer live inside a real
Megatron `MambaLayer` without rewriting the mixer math.

Why it exists: Megatron expects `[seq, batch, hidden]`, extra constructor
kwargs, and a `(output, bias)` return contract, while the MegaCpp POC recurrent layer
 expects a different config surface and `[batch, seq, hidden]` tensors.

What problem it solves: it preserves Megatron overlap and residual plumbing
while keeping the recurrence on a pure PyTorch path that is compatible with
regional compile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class WrapperConfigShim:
    """Read-through config shim matching the MegaCpp POC wrapper idea.

    Grounding:
    - MegaCpp POC source: `megatron_m2rnn.py::NanochatM2RNNMixer`
    """

    primary: Any
    secondary: Any

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.primary, name):
            return getattr(self.primary, name)
        return getattr(self.secondary, name)


class RegionalCompileM2RNNWrapper(torch.nn.Module):
    """Shape adapter extracted from the MegaCpp POC integration path.

    Grounded changes:
    - swallow Megatron-only construction kwargs instead of duplicating the
      recurrent layer signature
    - transpose `[seq, batch, hidden]` to `[batch, seq, hidden]` on ingress and
      reverse it on egress
    - return `(output, None)` so the surrounding Mamba block can keep its
      fused bias-dropout-add path unchanged
    """

    def __init__(
        self,
        recurrent_layer: torch.nn.Module,
        *,
        config: Any,
        model_config: Any | None = None,
        submodules: Any = None,
        layer_number: int | None = None,
        pg_collection: Any = None,
        pp_layer_offset: int = 0,
    ) -> None:
        super().__init__()
        del submodules, layer_number, pg_collection, pp_layer_offset
        self.recurrent_layer = recurrent_layer
        self.config = WrapperConfigShim(
            primary=model_config if model_config is not None else config,
            secondary=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_context: Any = None,
        packed_seq_params: Any = None,
    ) -> tuple[torch.Tensor, None]:
        del inference_context, packed_seq_params
        batch_first = hidden_states.transpose(0, 1).contiguous()
        output = self.recurrent_layer(batch_first)
        return output.transpose(0, 1).contiguous(), None


def compile_boundary_note() -> str:
    return (
        "The MegaCpp POC keeps this wrapper on a pure-PyTorch recurrence path so it can "
        "participate in regional_compile without depending on Triton kernel "
        "introspection or TMA-descriptor-specific behavior."
    )
