"""Public excerpt.

Source: MegaCpp public hybrid samples
Purpose: show how a compact pattern string expands into an explicit layer plan
Edited for clarity.
"""

BLOCK_KIND = {
    "A": "ablock",
    "M": "mblock",
    "E": "eblock",
    "R": "rblock",
    "C": "cblock",
}


def expand_hybrid_pattern(pattern: str) -> list[dict[str, object]]:
    layers = []
    for index, token in enumerate(pattern):
        kind = BLOCK_KIND.get(token)
        if kind is None:
            raise ValueError(f"unknown pattern token: {token}")
        layers.append(
            {
                "index": index,
                "token": token,
                "kind": kind,
                "uses_attention": token == "A",
                "uses_state_mixer": token == "M",
                "uses_expert_routing": token == "E",
            }
        )
    return layers


def count_block_kinds(pattern: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer in expand_hybrid_pattern(pattern):
        kind = layer["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts
