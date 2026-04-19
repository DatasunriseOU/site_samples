"""NAM56R pattern composition sample.

What it is: a public-safe receipt for how the short NAM56R pattern expands into
real block counts and special layer placements.

Why it exists: a compact pattern string like `AEMEAEMEAEMR` hides the actual
mix of A, E, M, and R families across depth 52.

What problem it solves: it gives one explicit block-count and placement summary
so public docs can talk about the model layout without forcing readers to
recompute the pattern by hand.
"""

from __future__ import annotations


PATTERN = "AEMEAEMEAEMR"
DEPTH = 52


def expand_pattern(pattern: str = PATTERN, depth: int = DEPTH) -> list[str]:
    return [pattern[i % len(pattern)] for i in range(depth)]


def summarize_pattern_composition(pattern: str = PATTERN, depth: int = DEPTH) -> dict[str, object]:
    expanded = expand_pattern(pattern, depth)
    counts = {symbol: expanded.count(symbol) for symbol in sorted(set(expanded))}
    a_ranks = [i for i, symbol in enumerate(expanded) if symbol == "A"]
    e_ranks = [i for i, symbol in enumerate(expanded) if symbol == "E"]
    m_ranks = [i for i, symbol in enumerate(expanded) if symbol == "M"]
    r_ranks = [i for i, symbol in enumerate(expanded) if symbol == "R"]
    return {
        "pattern": pattern,
        "depth": depth,
        "counts": counts,
        "a_layer_ranks": a_ranks,
        "e_layer_ranks": e_ranks,
        "m_layer_ranks": m_ranks,
        "r_layer_ranks": r_ranks,
        "notes": {
            "a_blocks": "attention family layers; selected ranks can switch from MLA to DSA or host Engram and mHC",
            "e_blocks": "expert family layers; these carry the MoE route",
            "m_blocks": "Mamba/MIMO family layers",
            "r_blocks": "recurrent family layers kept on an explicit index list",
        },
    }
