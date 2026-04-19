"""TPU backend fallback selection sample.

What it is: a public-safe receipt for how the MegaCpp POC picks between native
Pallas, Splash bridge, and plain fallback attention on TPU.

Why it exists: TPU attention features do not all fit one backend family, and
missing bridge symbols must degrade cleanly.

What problem it solves: it makes the fallback chain explicit so a failed native
kernel load does not silently claim support for masks or softcaps it cannot run.
"""

from __future__ import annotations


def choose_tpu_attention_path(
    *,
    xla_flash_available: bool,
    splash_available: bool,
    softcap: float,
    needs_local_window: bool,
) -> dict[str, object]:
    if xla_flash_available:
        if softcap > 0.0:
            return {
                "backend": "xla_flash_pallas_softcap",
                "native": True,
                "bridge": False,
                "reason": "native Pallas path handles fused softcap directly",
            }
        if needs_local_window:
            return {
                "backend": "xla_flash_pallas",
                "native": True,
                "bridge": False,
                "reason": "native Pallas path keeps local-window masking off the bridge",
            }
        return {
            "backend": "xla_flash_pallas",
            "native": True,
            "bridge": False,
            "reason": "plain causal TPU attention can stay on the native Pallas path",
        }
    if splash_available:
        return {
            "backend": "splash_call_jax",
            "native": False,
            "bridge": True,
            "reason": "Splash bridge is the fallback when native TPU flash symbols are missing",
        }
    return {
        "backend": "sdpa_fallback",
        "native": False,
        "bridge": False,
        "reason": "no TPU attention helper available; fall back to the generic attention path",
    }
