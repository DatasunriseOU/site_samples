"""FlexiDepth layer-skip adapter sample.

What it is: a public-safe excerpt of the MegaCpp POC FlexiDepth router and
adapter path.

Why it exists: FlexiDepth keeps static shapes while letting some tokens use a
lighter path instead of the full block.

What problem it solves: it shows how token-wise layer skipping can stay
compile-friendly by combining a router, a lightweight adapter, and a straight-
through hard mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlexiDepthConfigSample:
    n_embd: int
    router_reduction: int = 4
    adapter_reduction: int = 8
    threshold: float = 0.5


class FlexiDepthRouterSample(nn.Module):
    def __init__(self, config: FlexiDepthConfigSample) -> None:
        super().__init__()
        reduced = max(1, config.n_embd // config.router_reduction)
        self.down_proj = nn.Linear(config.n_embd, reduced, bias=False)
        self.norm = nn.RMSNorm(reduced)
        self.up_proj = nn.Linear(reduced, config.n_embd, bias=False)
        self.head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.down_proj(x))
        h = self.norm(h)
        h = self.up_proj(h)
        return torch.sigmoid(self.head(h))


class FlexiDepthAdapterSample(nn.Module):
    def __init__(self, config: FlexiDepthConfigSample) -> None:
        super().__init__()
        reduced = max(8, (4 * config.n_embd) // config.adapter_reduction)
        self.c_fc = nn.Linear(config.n_embd, reduced, bias=False)
        self.c_proj = nn.Linear(reduced, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


def flexidepth_mask(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    hard = (scores > threshold).to(dtype=scores.dtype)
    return hard - scores.detach() + scores


def apply_flexidepth_layer(
    x: torch.Tensor,
    full_block_output: torch.Tensor,
    router: FlexiDepthRouterSample,
    adapter: FlexiDepthAdapterSample,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = router(x)
    mask = flexidepth_mask(scores, threshold)
    adapter_out = adapter(F.rms_norm(x, (x.size(-1),)))
    out = mask * scores * full_block_output + (1.0 - mask) * (1.0 - scores) * adapter_out + x
    return out, scores
