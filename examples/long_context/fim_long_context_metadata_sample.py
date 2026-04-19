"""Long-context metadata permutation sample.

This example shows the exact metadata move that follows a fill-in-the-middle
transform. The problem it solves is scale: at long context the metadata arrays
are large enough that even a small alignment bug silently poisons supervision.
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


def permute_metadata_for_fim(
    meta_array: list[int],
    fim_result: FimResult,
    sentinel_value: int = 0,
) -> list[int]:
    if not fim_result.was_transformed:
        return meta_array

    split_start, split_end = fim_result.split_start, fim_result.split_end
    prefix_meta = meta_array[:split_start]
    middle_meta = meta_array[split_start:split_end]
    suffix_meta = meta_array[split_end:]

    if fim_result.is_spm:
        return [sentinel_value, sentinel_value] + suffix_meta + [sentinel_value] + prefix_meta + middle_meta + [sentinel_value]
    return [sentinel_value] + prefix_meta + [sentinel_value] + suffix_meta + [sentinel_value] + middle_meta + [sentinel_value]
