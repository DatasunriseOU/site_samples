"""Public-safe activation-offload target notes."""


def activation_offload_targets() -> list[str]:
    return [
        "long-context attention blocks when checkpointing is insufficient",
        "selected recurrent-state or mixer activations on tight-memory presets",
        "never everything by default; offload only enumerated hot surfaces",
    ]
