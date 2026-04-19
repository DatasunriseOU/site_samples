"""Long-context chunk boundary remap sample.

This example shows how chunk offsets are rewritten after a FIM transform. The
problem it solves is graph integrity: call and type edges become misleading if
their chunk spans no longer describe the current token order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FimResult:
    tokens: list[int]
    was_transformed: bool
    split_start: int = 0
    split_end: int = 0
    is_spm: bool = False


def remap_chunk_boundaries_for_fim(
    chunk_boundaries: list[dict],
    fim_result: FimResult,
    original_token_count: int,
) -> tuple[list[dict], list[int], list[int]]:
    if not fim_result.was_transformed or not chunk_boundaries:
        ordered = sorted(chunk_boundaries, key=lambda chunk: chunk["token_offset"])
        starts = [chunk["token_offset"] for chunk in ordered]
        ends = [ordered[idx + 1]["token_offset"] if idx + 1 < len(ordered) else original_token_count for idx in range(len(ordered))]
        return ordered, starts, ends

    split_start, split_end = fim_result.split_start, fim_result.split_end
    token_count = original_token_count
    if fim_result.is_spm:
        suffix_offset = 2
        prefix_offset = 2 + (token_count - split_end) + 1
        middle_offset = prefix_offset + split_start
    else:
        prefix_offset = 1
        suffix_offset = 1 + split_start + 1
        middle_offset = 1 + split_start + 1 + (token_count - split_end) + 1

    ordered = sorted(chunk_boundaries, key=lambda chunk: int(chunk.get("token_offset", 0)))
    ends_orig = [ordered[idx + 1]["token_offset"] if idx + 1 < len(ordered) else token_count for idx in range(len(ordered))]

    remapped: list[dict] = []
    starts: list[int] = []
    ends: list[int] = []
    for chunk, chunk_end in zip(ordered, ends_orig):
        chunk_start = int(chunk.get("token_offset", 0))
        if chunk_end <= split_start:
            new_start = prefix_offset + chunk_start
            new_end = prefix_offset + chunk_end
        elif chunk_start >= split_end:
            new_start = suffix_offset + (chunk_start - split_end)
            new_end = suffix_offset + (chunk_end - split_end)
        elif chunk_start >= split_start and chunk_end <= split_end:
            new_start = middle_offset + (chunk_start - split_start)
            new_end = middle_offset + (chunk_end - split_start)
        else:
            continue

        new_chunk = dict(chunk)
        new_chunk["token_offset"] = new_start
        remapped.append(new_chunk)
        starts.append(new_start)
        ends.append(new_end)

    return remapped, starts, ends
