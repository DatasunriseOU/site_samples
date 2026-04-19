"""Platform embedding sample.

What it is: a public-safe sample for document-level platform label IDs used by
MegaCpp POC enriched training data.

Why it exists: some documents carry platform context such as OS, compiler, or
hardware tags that apply to the whole sample rather than one token.

What problem it solves: it shows how per-document platform labels are turned
into one broadcast-ready embedding input instead of being mixed into token text.
"""

from __future__ import annotations


def summarize_platform_ids(platform_ids: list[list[int]]) -> dict[str, object]:
    non_empty = [row for row in platform_ids if row]
    max_labels = max((len(row) for row in non_empty), default=0)
    return {
        "documents": len(platform_ids),
        "documents_with_platform_labels": len(non_empty),
        "max_labels_per_document": max_labels,
        "embedding_mode": "sum",
        "padding_id": 0,
        "broadcast_shape": "(B, 1, n_embd)",
    }
