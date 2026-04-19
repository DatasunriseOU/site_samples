"""NAM56R block taxonomy sample.

What it is: a public-safe summary of the real NAM56R block letters and what
kind of layer each one stands for.

Why it exists: the pattern string is short, but the runtime consequences are
not obvious unless the letters are decoded into real block families.

What problem it solves: it gives one place where readers can see how A, E, M,
and R blocks divide the model into attention, MoE, Mamba, and custom recurrent
surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass


NAM56R_PATTERN = "AEMEAEMEAEMR"
NAM56R_DEPTH = 52


@dataclass(frozen=True)
class BlockDefinition:
    symbol: str
    name: str
    role: str
    primary_compute: str
    typical_features: tuple[str, ...]


BLOCK_DEFINITIONS = {
    "A": BlockDefinition(
        symbol="A",
        name="A-block",
        role="attention block",
        primary_compute="dense or selective attention",
        typical_features=("MLA", "DSA on selected A-layer ranks", "Engram", "mHC"),
    ),
    "E": BlockDefinition(
        symbol="E",
        name="E-block",
        role="expert feed-forward block",
        primary_compute="Mixture-of-Experts",
        typical_features=("16 routed relu2 experts", "1 shared SwiGLU expert", "MoD on E-blocks"),
    ),
    "M": BlockDefinition(
        symbol="M",
        name="M-block",
        role="Mamba-3 state-space block",
        primary_compute="selective state-space mixing",
        typical_features=("Mamba-3", "MIMO path", "SSM-specific optimizer grouping"),
    ),
    "R": BlockDefinition(
        symbol="R",
        name="R-block",
        role="custom recurrent block",
        primary_compute="M2RNN-style recurrent mixer",
        typical_features=("custom runtime seam", "not Megatron-native", "pattern-preserved custom indices"),
    ),
}


def expand_nam56r_pattern(*, pattern: str = NAM56R_PATTERN, depth: int = NAM56R_DEPTH) -> list[str]:
    return [pattern[i % len(pattern)] for i in range(depth)]


def summarize_block_taxonomy(*, pattern: str = NAM56R_PATTERN, depth: int = NAM56R_DEPTH) -> dict[str, object]:
    expanded = expand_nam56r_pattern(pattern=pattern, depth=depth)
    counts: dict[str, int] = {}
    for symbol in expanded:
        counts[symbol] = counts.get(symbol, 0) + 1
    return {
        "pattern": pattern,
        "depth": depth,
        "counts": counts,
        "definitions": {
            symbol: {
                "name": info.name,
                "role": info.role,
                "primary_compute": info.primary_compute,
                "typical_features": list(info.typical_features),
            }
            for symbol, info in BLOCK_DEFINITIONS.items()
        },
    }
