"""MegaCpp POC example: do not confuse driver-visible signals with runtime proof.

What this solves in simple words:
- capability tables, helper cubins, and signed metadata can all make a path
  look real before the hardware path is actually proven;
- the publication contract should separate hints, routing evidence, and
  end-to-end execution receipts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilitySignal:
    name: str
    indicates: str
    is_runtime_proof: bool


SIGNALS = (
    CapabilitySignal(
        name="compute-capability table patch",
        indicates="the driver can be nudged to route GB10 as if it were a different target",
        is_runtime_proof=False,
    ),
    CapabilitySignal(
        name="helper cubin present in libcuda",
        indicates="the software stack knows about a richer path",
        is_runtime_proof=False,
    ),
    CapabilitySignal(
        name=".nv.capmerc signed capability metadata",
        indicates="the driver protects a capability boundary cryptographically",
        is_runtime_proof=False,
    ),
    CapabilitySignal(
        name="baseline sm_100a arithmetic cubin completed on GB10",
        indicates="some sm_100a SASS executes on GB10 after arch rewriting",
        is_runtime_proof=False,
    ),
    CapabilitySignal(
        name="exact tcgen05 kernel completed and produced expected output",
        indicates="the claimed path was proven end to end",
        is_runtime_proof=True,
    ),
)


def non_proof_signals() -> tuple[CapabilitySignal, ...]:
    return tuple(signal for signal in SIGNALS if not signal.is_runtime_proof)


def publication_verdict(signals: tuple[CapabilitySignal, ...]) -> str:
    if any(signal.is_runtime_proof for signal in signals):
        return "Runtime proof exists for the exact path."
    return (
        "Interesting driver-visible evidence exists, but publication should stop "
        "at software routing/gating claims until the exact path completes."
    )


if __name__ == "__main__":
    print(publication_verdict(non_proof_signals()))
