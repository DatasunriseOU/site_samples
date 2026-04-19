"""Public hybrid architecture sample for article references.

This module keeps the layer-pattern contract explicit without exposing
private paths, internal identifiers, or repo-specific entrypoints.
"""

BLOCK_KIND = {
    "A": "ablock",
    "M": "mblock",
    "E": "eblock",
    "R": "rblock",
    "C": "cblock",
}

FULL_PATTERN = "AEMEAEMEAEMR"


def expand_pattern(pattern: str) -> list[dict[str, object]]:
    layers = []
    for index, token in enumerate(pattern):
        kind = BLOCK_KIND.get(token)
        if kind is None:
            raise ValueError(f"unknown pattern token: {token}")
        layers.append({"index": index, "token": token, "kind": kind})
    return layers


def summarize_pattern(pattern: str) -> dict[str, object]:
    layers = expand_pattern(pattern)
    counts: dict[str, int] = {}
    for layer in layers:
        kind = layer["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "pattern": pattern,
        "depth": len(layers),
        "counts": counts,
        "layers": layers,
    }


def build_hybrid_stack(pattern: str = FULL_PATTERN) -> list[dict[str, object]]:
    stack = []
    for layer in expand_pattern(pattern):
        stack.append(
            {
                "name": f"layer_{layer['index']:02d}",
                "block_kind": layer["kind"],
                "uses_attention": layer["token"] == "A",
                "uses_state_mixer": layer["token"] == "M",
                "uses_routing": layer["token"] == "E",
            }
        )
    return stack
