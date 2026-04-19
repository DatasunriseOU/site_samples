"""Packed-row construction example.

This shows how tokenized documents are packed into fixed-length training rows.
The problem it solves is wasted context: instead of training on one short file
per row, the packer fills each row with as many documents as fit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedDoc:
    token_ids: list[int]
    doc_id: int

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def pack_documents(docs: list[NormalizedDoc], *, seq_len: int) -> list[dict[str, object]]:
    """Pack tokenized docs into fixed-length rows with per-token doc ids."""
    rows: list[dict[str, object]] = []
    current_tokens: list[int] = []
    current_doc_ids: list[int] = []

    for doc in docs:
        if doc.token_count > seq_len:
            continue
        if len(current_tokens) + doc.token_count > seq_len:
            rows.append(_finalize_row(current_tokens, current_doc_ids, seq_len))
            current_tokens = []
            current_doc_ids = []
        current_tokens.extend(doc.token_ids)
        current_doc_ids.extend([doc.doc_id] * doc.token_count)

    if current_tokens:
        rows.append(_finalize_row(current_tokens, current_doc_ids, seq_len))
    return rows


def _finalize_row(token_ids: list[int], doc_ids: list[int], seq_len: int) -> dict[str, object]:
    pad = seq_len - len(token_ids)
    return {
        "input_ids": token_ids + [0] * pad,
        "doc_ids": doc_ids + [-1] * pad,
        "valid_token_count": len(token_ids),
        "num_docs": len({doc_id for doc_id in doc_ids if doc_id >= 0}),
        "loss_mask": [1] * len(token_ids) + [0] * pad,
    }
