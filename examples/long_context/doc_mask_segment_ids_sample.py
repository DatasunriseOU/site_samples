"""Document-mask segment ID sample.

What it is: a public-safe sample for long-context segment IDs and document IDs
that survive augmentation and packing.

Why it exists: long-context runs often pack multiple documents together, and
stateful blocks need a reliable signal for where one document ends and the next
starts.

What problem it solves: it keeps document masking and state reset surfaces tied
to explicit per-token IDs instead of relying on guessed boundaries.
"""

from __future__ import annotations


def build_segment_id_receipt(doc_ids: list[int]) -> dict[str, object]:
    if not doc_ids:
        return {"doc_ids": [], "segment_ids": [], "num_segments": 0}
    segment_ids: list[int] = []
    current_segment = 0
    prev_doc = doc_ids[0]
    for doc_id in doc_ids:
        if doc_id != prev_doc:
            current_segment += 1
            prev_doc = doc_id
        segment_ids.append(current_segment)
    return {
        "doc_ids": doc_ids,
        "segment_ids": segment_ids,
        "num_segments": current_segment + 1,
        "uses": ["document masking", "SSM state reset", "packed-row boundaries"],
    }
