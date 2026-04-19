"""Public-safe DASH example: periodic, AdamW-safe, post-backward/pre-step."""


def dash_plan(*, step: int, interval: int = 2000, optimizer_family: str = "adamw") -> dict[str, object]:
    return {
        "apply_now": optimizer_family == "adamw" and step % interval == 0,
        "placement": "post-backward-pre-step",
        "optimizer_family": optimizer_family,
        "interval": interval,
    }
