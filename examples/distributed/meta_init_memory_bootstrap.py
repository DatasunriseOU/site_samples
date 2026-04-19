"""Public-safe meta-init bootstrap excerpt.

This mirrors the MegaCpp POC pattern used in the internal meta-init helper:
instantiate on the meta device, materialize empty tensors directly on the
target device, then run the model's initializer once real storage exists.
"""

from __future__ import annotations

from typing import Any


def create_model_on_device(model_cls: type, config: Any, device: Any):
    """Create a model without transient full-size CPU parameter storage.

    The MegaCpp POC-backed bootstrap sequence is:
    1. Instantiate on the meta device so parameters have metadata only.
    2. Call ``to_empty(device=...)`` to materialize empty tensors directly on
       the destination device.
    3. Run ``init_weights()`` after materialization so no parameter remains on
       the meta device.
    """

    import torch

    with torch.device("meta"):
        model = model_cls(config)
    model = model.to_empty(device=device)
    model.init_weights()
    return model


def bootstrap_summary() -> list[str]:
    return [
        "Instantiate on the meta device first so tensors carry shape metadata without backing storage.",
        "Materialize empty tensors directly on the destination device with to_empty(device=...).",
        "Run init_weights() only after materialization so no parameter remains on the meta device.",
    ]
