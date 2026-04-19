"""FlexiDepth skip-router sample.

What it is: a public-safe summary of the adaptive layer-skip router used for
token-wise full-path versus cheap-path selection.

Why it exists: adaptive skipping only works if the router, adapter, and skip
loss all agree on one static-shape contract.

What problem it solves: it makes the skip policy explicit so readers can see
why KV is preserved, why the adapter exists, and how the skip loss is formed.
"""

from __future__ import annotations


def flexidepth_router_shape(n_embd: int, *, router_reduction: int) -> dict[str, int]:
    """Return the grounded bottleneck sizes for the FlexiDepth router."""

    reduced = max(1, n_embd // router_reduction)
    return {
        "input_dim": n_embd,
        "router_hidden_dim": reduced,
        "head_dim": 1,
    }


def flexidepth_path_contract() -> tuple[str, ...]:
    """Summarize the MegaCpp POC skip-path policy without private module glue."""

    return (
        "tokens above threshold use the full attention plus full FFN path",
        "tokens below threshold keep KV-compatible attention bookkeeping",
        "low-score tokens use a cheap adapter instead of the full FFN",
        "both paths stay static-shape friendly and are mixed with elementwise masks",
    )


def flexidepth_skip_loss(scores_per_layer: list[float]) -> float:
    """Public-safe scalar version of the squared total-usage penalty."""

    if not scores_per_layer:
        return 0.0
    total = sum(scores_per_layer)
    return (total * total) / len(scores_per_layer)
