"""MegaCpp POC-based n-gram hash embedding example.

This feature enriches token embeddings with cheap hashed local token-pattern
lookups. It exists so the model can keep repeated code motifs in a compact
memory-like side channel instead of inflating the main embedding stack.

The sample keeps the real design shape from the MegaCpp POC:
- one unified embedding table for all hash heads;
- vectorized hash construction for every table at once;
- an optional CPU-offload mode for the table lookup path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMES = [
    499_801, 499_819, 499_853, 499_879, 499_883, 499_897, 499_903, 499_927,
    499_943, 499_957, 499_969, 499_973, 499_979, 500_009, 500_029, 500_041,
]


def pick_primes(count: int, target_size: int) -> list[int]:
    candidates = [prime for prime in PRIMES if abs(prime - target_size) / target_size < 0.5]
    if len(candidates) >= count:
        return candidates[:count]
    return [target_size + i for i in range(count)]


class EmbeddingTableView:
    """Expose unified-table slices through the old per-table access pattern."""

    __slots__ = ("_unified_weight", "_start", "_end")

    def __init__(self, unified_weight: torch.nn.Parameter, start: int, end: int) -> None:
        self._unified_weight = unified_weight
        self._start = start
        self._end = end

    @property
    def weight(self) -> torch.Tensor:
        return self._unified_weight[self._start:self._end]


class NgramHashEmbeddingSample(nn.Module):
    """Hash local token patterns into one shared embedding surface."""

    def __init__(
        self,
        n_embd: int,
        orders: tuple[int, ...] = (2, 3),
        num_heads: int = 8,
        table_size: int = 500_000,
        embed_dim: int = 0,
        dropout: float = 0.0,
        offload: bool = False,
    ) -> None:
        super().__init__()
        if not orders:
            raise ValueError("orders must contain at least one n-gram order")

        self.n_embd = n_embd
        self.orders = tuple(orders)
        self.num_heads = num_heads
        self.num_tables = len(self.orders) * num_heads
        self.offload = offload
        self.embed_dim = embed_dim if embed_dim > 0 else max(16, n_embd // self.num_tables)
        self.max_order = max(self.orders)

        self.table_sizes = pick_primes(self.num_tables, table_size)
        offsets: list[int] = []
        total = 0
        for size in self.table_sizes:
            offsets.append(total)
            total += size

        self.register_buffer("table_offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("table_sizes_t", torch.tensor(self.table_sizes, dtype=torch.long))
        self.unified_table = nn.Embedding(total, self.embed_dim)

        mults = torch.randint(1, 2**31, (self.num_tables, self.max_order), dtype=torch.long)
        self.register_buffer("hash_mults", mults | 1)
        self.register_buffer("hash_bias", torch.randint(0, 2**31, (self.num_tables,), dtype=torch.long))

        order_list: list[int] = []
        for order in self.orders:
            for _ in range(self.num_heads):
                order_list.append(order)
        order_mask = torch.zeros(self.max_order, self.num_tables, dtype=torch.long)
        for table_idx, order in enumerate(order_list):
            for position in range(order):
                order_mask[position, table_idx] = 1
        self.register_buffer("order_mask", order_mask)

        self.out_proj = nn.Linear(self.num_tables * self.embed_dim, n_embd, bias=False)
        torch.nn.init.zeros_(self.out_proj.weight)
        self.dropout = nn.Dropout(dropout)
        self.tables = [
            EmbeddingTableView(self.unified_table.weight, start, start + size)
            for start, size in zip(offsets, self.table_sizes)
        ]

    def _hash_all(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, steps = token_ids.shape
        shifted = torch.zeros(self.max_order, batch, steps, dtype=torch.long, device=token_ids.device)
        shifted[0] = token_ids
        for shift in range(1, self.max_order):
            shifted[shift, :, shift:] = token_ids[:, :-shift]

        mults = self.hash_mults.t().unsqueeze(-1).unsqueeze(-1)
        mask = self.order_mask.unsqueeze(-1).unsqueeze(-1)
        product = (mults * shifted.unsqueeze(1)) * mask

        hashed = product[0]
        for idx in range(1, self.max_order):
            hashed = hashed ^ product[idx]

        hashed = hashed ^ self.hash_bias.unsqueeze(-1).unsqueeze(-1)
        hashed = hashed % self.table_sizes_t.unsqueeze(-1).unsqueeze(-1)
        unified_indices = hashed + self.table_offsets.unsqueeze(-1).unsqueeze(-1)
        return unified_indices.permute(1, 0, 2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        target_device = token_ids.device
        if self.offload:
            token_ids = token_ids.cpu()

        unified_indices = self._hash_all(token_ids)
        batch, num_tables, steps = unified_indices.shape
        flat_indices = unified_indices.reshape(-1)
        flat_emb = F.embedding(flat_indices, self.unified_table.weight)
        emb = flat_emb.view(batch, num_tables, steps, self.embed_dim)
        emb = emb.permute(0, 2, 1, 3).reshape(batch, steps, num_tables * self.embed_dim)

        if self.offload:
            emb = emb.to(device=target_device, dtype=self.out_proj.weight.dtype)

        out = self.out_proj(emb)
        return self.dropout(out)
