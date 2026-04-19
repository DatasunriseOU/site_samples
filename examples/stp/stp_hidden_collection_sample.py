"""STP hidden-state collection sample.

What it is: a public-safe receipt for how STP collects hidden states from the
last layer or from a configured set of intermediate layers.

Why it exists: STP has both single-layer and multi-layer variants, and the
collection path must make that explicit.

What problem it solves: it shows how the training loop decides whether to use
the last hidden state only or accumulate a multi-layer list for averaged STP
loss.
"""

from __future__ import annotations

from collections.abc import Iterable


def collect_stp_hidden_states(
    *,
    current_layer: int,
    reduced_hidden: object,
    configured_layers: Iterable[int] | None,
    collected: list[object],
) -> list[object]:
    """Append reduced hidden states when the current layer is part of the STP set."""

    if configured_layers is None:
        return collected
    stp_layer_set = set(configured_layers)
    if current_layer in stp_layer_set:
        collected.append(reduced_hidden)
    return collected
