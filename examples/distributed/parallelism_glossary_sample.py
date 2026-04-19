"""Grounded glossary for distributed split names."""

GLOSSARY = {
    "dp": ("replicate parameters, split batches", "does not reduce per-layer activation pressure"),
    "tp": ("split tensor dimensions across ranks", "does not solve pipeline residency or expert imbalance"),
    "pp": ("split layer stacks into stages", "does not make activation lifetime free; schedule still matters"),
    "cp": ("split long-context ownership across ranks", "does not replace TP/PP for parameter memory"),
    "sp": ("split sequence work inside TP regions", "does not by itself fix expert or optimizer pressure"),
    "ep": ("split expert ownership across ranks", "does not solve router hotspots or dense-path memory"),
}


def describe_parallelism(mode: str) -> dict[str, str]:
    key = mode.lower()
    if key not in GLOSSARY:
        raise KeyError(f"unknown mode: {mode}")
    ownership, limitation = GLOSSARY[key]
    return {"mode": key, "ownership": ownership, "does_not_solve": limitation}
