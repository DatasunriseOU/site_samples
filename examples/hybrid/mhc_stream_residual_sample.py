"""mHC stream and residual interaction sample grounded in the MegaCpp POC GPT config.

What it is: a public-safe summary of the knobs that change cross-layer mixing
and residual behavior around mHC-enabled layers.

Why it exists: mHC does not replace every residual rule. It adds multi-stream
mixing, while other switches such as FP32 residual and AttnRes still affect how
the hidden state is carried.

What problem it solves: it shows the real mutually-exclusive or cooperating
surfaces so readers do not assume there is one generic "residual mode" flag.
"""

from __future__ import annotations


def summarize_mhc_and_residual_surface(config: object) -> dict[str, object]:
    mhc_enabled = bool(getattr(config, "mhc_enabled", False))
    n_streams = int(getattr(config, "mhc_n_streams", 1))
    attn_res_enabled = bool(getattr(config, "attn_res_enabled", False))
    fp32_residual = bool(getattr(config, "fp32_residual", False))

    return {
        "mhc": {
            "enabled": mhc_enabled,
            "n_streams": n_streams,
            "dynamic": bool(getattr(config, "mhc_dynamic", False)),
            "dynamic_mode": str(getattr(config, "mhc_dynamic_mode", "maxtext")),
            "sinkhorn_iters": int(getattr(config, "mhc_sinkhorn_iters", 5)),
            "blend_alpha": float(getattr(config, "mhc_blend_alpha", 1.0)),
        },
        "residual": {
            "fp32_residual": fp32_residual,
            "attn_res_enabled": attn_res_enabled,
            "attn_res_allowed_with_mhc": not (mhc_enabled and n_streams > 1 and attn_res_enabled),
            "m2rnn_use_residual": bool(getattr(config, "m2rnn_use_residual", True)),
        },
        "explanation": {
            "mhc": "mixes multiple hidden-state streams between layers instead of using only one running stream",
            "fp32_residual": "keeps residual adds in float32 to reduce bf16 drift on long runs",
            "attn_res": "learned cross-layer residual attention; disabled when multi-stream hyper-connections already own that job",
        },
    }

