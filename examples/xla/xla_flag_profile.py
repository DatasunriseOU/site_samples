"""Public-safe TPU/XLA configuration sample used in article references."""

XLA_PROFILE = {
    "precision": "bf16",
    "compile_cache": "enabled",
    "spmd_mode": True,
    "shape_guard": "strict",
}


def summarize_xla_profile(profile: dict) -> str:
    enabled = "on" if profile.get("spmd_mode") else "off"
    return f"precision={profile['precision']} spmd={enabled}"
