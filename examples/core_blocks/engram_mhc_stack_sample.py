"""MegaCpp POC-based A-block feature wiring example.

This sample shows how the core side features are combined on attention blocks:
the token path may get n-gram hash enrichment up front, the attention block may
add an Engram side branch, and mHC then mixes the active branch set back into a
single residual stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .engram_branch_sample import EngramBranchSample
from .mhc_branch_mixer_sample import ManifoldBranchMixerSample
from .ngram_hash_embedding_sample import NgramHashEmbeddingSample


@dataclass
class CoreBlockFeatureFlags:
    use_ngram_hash: bool = False
    use_engram: bool = False
    use_mhc: bool = False


class CoreABlockFeatureStackSample(nn.Module):
    """Minimal integration contract for the core block-side feature stack."""

    def __init__(self, n_embd: int, *, flags: CoreBlockFeatureFlags) -> None:
        super().__init__()
        self.flags = flags
        self.ngram_hash = NgramHashEmbeddingSample(n_embd=n_embd, orders=(2, 3), num_heads=2) if flags.use_ngram_hash else None
        self.engram = EngramBranchSample(n_embd=n_embd, ngram_orders="2,3", gated=True, conv_kernel=4) if flags.use_engram else None
        self.mhc = ManifoldBranchMixerSample(n_embd=n_embd, sinkhorn_iters=1) if flags.use_mhc else None

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = token_embeddings
        if self.ngram_hash is not None:
            hidden = hidden + self.ngram_hash(token_ids)

        branches = [hidden]
        if self.engram is not None:
            branches.append(self.engram(hidden, doc_ids=doc_ids))

        if self.mhc is None or len(branches) < 2:
            return branches[0]
        return self.mhc(branches)
