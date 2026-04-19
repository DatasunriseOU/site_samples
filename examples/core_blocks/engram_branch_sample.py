"""MegaCpp POC-based Engram branch example.

This feature adds a small causal n-gram side branch to attention blocks. The
goal is to keep short local code motifs available without paying for another
full attention pass.

The sample keeps the real contract from the MegaCpp POC:
- pooled n-gram features are built causally;
- optional packed `doc_ids` stop pooling across document boundaries;
- the upgraded mode adds sigmoid gating and a small causal depthwise conv.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_ngram_orders(ngram_orders: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(ngram_orders, str):
        parts = [x.strip() for x in ngram_orders.split(",")]
        orders = [int(x) for x in parts if x]
    else:
        orders = [int(x) for x in ngram_orders]

    deduped: list[int] = []
    seen: set[int] = set()
    for order in orders:
        if order <= 0 or order in seen:
            continue
        deduped.append(order)
        seen.add(order)
    return tuple(deduped or [2, 3, 4])


class RMSNormNoWeight(nn.Module):
    """Minimal RMSNorm used by the Engram gate path."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps)


class EngramBranchSample(nn.Module):
    """Causal n-gram branch extracted from the MegaCpp POC Engram path."""

    def __init__(
        self,
        n_embd: int,
        ngram_orders: str | tuple[int, ...] = "2,3,4",
        bottleneck_dim: int = 0,
        dropout: float = 0.0,
        gated: bool = False,
        gate_sqrt_compress: bool = False,
        conv_kernel: int = 0,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.ngram_orders = parse_ngram_orders(ngram_orders)
        self.bottleneck_dim = int(bottleneck_dim) if bottleneck_dim else max(1, n_embd // 4)
        self.gated = gated
        self.gate_sqrt_compress = bool(gate_sqrt_compress)
        self.conv_kernel = conv_kernel

        self.in_proj = nn.Linear(n_embd, self.bottleneck_dim, bias=False)
        self.order_mix = nn.ModuleList(
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim, bias=False)
            for _ in self.ngram_orders
        )
        self.dropout = nn.Dropout(dropout)

        if gated:
            self.gate_key_proj = nn.Linear(self.bottleneck_dim, n_embd, bias=False)
            self.gate_norm_h = RMSNormNoWeight(n_embd)
            self.gate_norm_k = RMSNormNoWeight(n_embd)
            self.value_proj = nn.Linear(self.bottleneck_dim, n_embd, bias=False)
            torch.nn.init.zeros_(self.value_proj.weight)
        else:
            self.out_proj = nn.Linear(self.bottleneck_dim, n_embd, bias=False)
            torch.nn.init.zeros_(self.out_proj.weight)

        if conv_kernel > 0:
            self.conv_norm = RMSNormNoWeight(n_embd)
            self.conv_weight = nn.Parameter(torch.empty(n_embd, 1, conv_kernel))
            torch.nn.init.normal_(self.conv_weight, std=0.02)

    def _same_doc_shift_mask(self, doc_ids: torch.Tensor, shift: int, dtype: torch.dtype) -> torch.Tensor:
        batch, steps = doc_ids.shape
        if shift == 0:
            return torch.ones(batch, steps, device=doc_ids.device, dtype=dtype)
        if shift >= steps:
            return torch.zeros(batch, steps, device=doc_ids.device, dtype=dtype)
        return F.pad((doc_ids[:, shift:] == doc_ids[:, :-shift]).to(dtype), (shift, 0))

    def _validate_doc_ids(self, x: torch.Tensor, doc_ids: torch.Tensor) -> None:
        if doc_ids.ndim != 2:
            raise ValueError(f"doc_ids must have shape (B, T); got {tuple(doc_ids.shape)}")
        if doc_ids.shape != x.shape[:2]:
            raise ValueError(
                "doc_ids must match x batch/sequence dimensions: "
                f"x.shape[:2]={tuple(x.shape[:2])}, doc_ids.shape={tuple(doc_ids.shape)}"
            )

    def _causal_local_average(
        self,
        x: torch.Tensor,
        order: int,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if doc_ids is not None:
            total = torch.zeros_like(x)
            for shift in range(order):
                if shift == 0:
                    shifted = x
                elif shift < x.shape[1]:
                    shifted = F.pad(x[:, :-shift, :], (0, 0, shift, 0))
                    shifted = shifted * self._same_doc_shift_mask(doc_ids, shift, x.dtype).unsqueeze(-1)
                else:
                    continue
                total = total + shifted
            return total / order

        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (order - 1, 0))
        x_t = F.avg_pool1d(x_t, kernel_size=order, stride=1)
        return x_t.transpose(1, 2)

    def _manual_depthwise_causal_conv(
        self,
        ct: torch.Tensor,
        conv_weight: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = ct.shape[2]
        kernel = conv_weight.shape[1]
        result = torch.zeros(ct.shape[0], ct.shape[1], steps, device=ct.device, dtype=ct.dtype)
        for shift in range(kernel):
            if shift == 0:
                shifted = ct
            elif shift < steps:
                shifted = F.pad(ct[:, :, :-shift], (shift, 0))
                if doc_ids is not None:
                    shifted = shifted * self._same_doc_shift_mask(doc_ids, shift, ct.dtype).unsqueeze(1)
            else:
                continue
            weight_idx = kernel - 1 - shift
            result = result + shifted * conv_weight[:, weight_idx].view(1, -1, 1)
        return result

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        if doc_ids is not None:
            self._validate_doc_ids(x, doc_ids)

        z = self.in_proj(x)
        y = torch.zeros_like(z)
        for order, mix in zip(self.ngram_orders, self.order_mix):
            local = self._causal_local_average(z, order, doc_ids=doc_ids)
            y = y + mix(local)
        y = y / len(self.ngram_orders)

        if self.gated:
            k = self.gate_key_proj(y)
            h_norm = self.gate_norm_h(x)
            k_norm = self.gate_norm_k(k)
            gate_logits = (h_norm * k_norm).sum(dim=-1, keepdim=True) / math.sqrt(self.n_embd)
            if self.gate_sqrt_compress:
                gate_logits = torch.sign(gate_logits) * torch.sqrt(gate_logits.abs().clamp_min(1e-6))
            out = torch.sigmoid(gate_logits) * self.value_proj(y)
        else:
            out = self.out_proj(y)

        if self.conv_kernel > 0:
            normed = self.conv_norm(out)
            ct = normed.transpose(1, 2)
            windows = self._manual_depthwise_causal_conv(ct, self.conv_weight[:, 0, :], doc_ids=doc_ids)
            out = F.silu(windows).transpose(1, 2)

        return self.dropout(out)
