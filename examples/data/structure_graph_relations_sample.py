"""Structure graph relation sample.

What it is: a public-safe excerpt of the chunk-relation metadata used by the
MegaCpp POC structure graph enricher and relation bias.

Why it exists: code chunks can be related through calls, type dependencies, and
shared dependency depth, and those relations have to be normalized into stable
relation IDs.

What problem it solves: it shows how chunk metadata becomes graph-ready model
inputs instead of staying as loose JSON records.
"""

from __future__ import annotations


RELATION_TYPE_IDS = {
    "same_chunk": 0,
    "caller_callee": 1,
    "callee_caller": 2,
    "type_dependency": 3,
    "type_dependent": 4,
    "same_dep_level": 5,
    "adjacent_dep_level": 6,
    "preamble_to_code": 7,
    "code_to_preamble": 8,
}


def build_chunk_relation_summary(
    *,
    chunk_dep_levels: list[int],
    call_edges: list[tuple[int, int]],
    type_edges: list[tuple[int, int]],
) -> dict[str, object]:
    adjacent_pairs = []
    for src, src_depth in enumerate(chunk_dep_levels):
        for dst, dst_depth in enumerate(chunk_dep_levels):
            if abs(int(src_depth) - int(dst_depth)) == 1:
                adjacent_pairs.append((src, dst))

    return {
        "relation_type_ids": RELATION_TYPE_IDS,
        "call_edges": [{"src": int(src), "dst": int(dst)} for src, dst in call_edges],
        "type_edges": [{"src": int(src), "dst": int(dst)} for src, dst in type_edges],
        "same_dep_level_pairs": [
            {"src": i, "dst": j}
            for i, lhs in enumerate(chunk_dep_levels)
            for j, rhs in enumerate(chunk_dep_levels)
            if i != j and int(lhs) == int(rhs)
        ],
        "adjacent_dep_level_pairs": [
            {"src": src, "dst": dst} for src, dst in adjacent_pairs
        ],
    }
