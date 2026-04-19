"""FlexiDepth frozen-model adapter wiring sample.

What it is: a public-safe excerpt of the MegaCpp POC-style FlexiDepth frozen
backbone contract.

Why it exists: FlexiDepth is meant to be added to a pretrained model without
retraining the whole block stack. The router and the lightweight adapter are
the parts meant to move first while the base block can stay frozen.

What problem it solves: it makes the frozen-backbone plus trainable-adapter
split explicit, which is the main deployment story for FlexiDepth.
"""

from __future__ import annotations

import torch.nn as nn


def freeze_flexidepth_backbone_sample(block: nn.Module, router: nn.Module, adapter: nn.Module) -> dict[str, int]:
    frozen = 0
    trainable = 0
    for parameter in block.parameters():
        parameter.requires_grad = False
        frozen += parameter.numel()
    for module in (router, adapter):
        for parameter in module.parameters():
            parameter.requires_grad = True
            trainable += parameter.numel()
    return {
        "frozen_backbone_params": frozen,
        "trainable_router_adapter_params": trainable,
    }


def describe_flexidepth_frozen_wiring_sample() -> dict[str, str]:
    return {
        "frozen_path": "full transformer block stays fixed and still produces the full-path delta",
        "trainable_path": "router learns token-wise skip scores and adapter learns the cheap fallback path",
        "runtime_contract": "both paths keep static shapes, then the router mask selects which contribution enters the residual",
    }
