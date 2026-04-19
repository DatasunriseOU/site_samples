"""Public hybrid pattern example for article references."""

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
