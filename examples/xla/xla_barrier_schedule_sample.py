"""XLA graph barrier schedule sample.

What it is: a public-safe excerpt of the MegaCpp POC forward/backward barrier
schedule used to split giant TPU graphs into smaller compile units.

Why it exists: large hybrid models can hang or over-allocate during TPU compile
when the whole forward and backward stay in one graph.

What problem it solves: it documents when `mark_step()` barriers are inserted so
TPU graph splitting stays intentional rather than accidental.
"""

from __future__ import annotations

from collections.abc import Iterable


def barrier_layers(*, num_layers: int, barrier_every_n_layers: int) -> list[int]:
    if barrier_every_n_layers <= 0:
        return []
    return [
        layer_idx
        for layer_idx in range(1, num_layers + 1)
        if layer_idx % barrier_every_n_layers == 0 and layer_idx < num_layers
    ]


def build_xla_barrier_schedule(*, num_layers: int, barrier_every_n_layers: int) -> dict[str, object]:
    forward_boundaries = barrier_layers(
        num_layers=num_layers,
        barrier_every_n_layers=barrier_every_n_layers,
    )
    return {
        "num_layers": num_layers,
        "barrier_every_n_layers": barrier_every_n_layers,
        "forward_mark_step_after_layers": forward_boundaries,
        "backward_hook_boundaries": list(forward_boundaries),
        "notes": [
            "Barriers are only useful on XLA devices.",
            "Backward hooks mirror the forward split so the backward graph does not stay monolithic.",
        ],
    }


def describe_xla_barrier_policy(*, barrier_every_n_layers: int) -> str:
    if barrier_every_n_layers <= 0:
        return "No explicit XLA barriers: keep the model in one compiled graph."
    return (
        "Insert mark_step barriers every "
        f"{barrier_every_n_layers} layers to split the TPU graph into smaller compile units."
    )
