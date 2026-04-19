"""Splash Local mask cache sample.

What it is: a public-safe excerpt of the MegaCpp POC mask-composition cache used
for Splash Local and related TPU sparse-local paths.

Why it exists: recreating logically identical masks with new Python identities
forces unnecessary TPU recompiles.

What problem it solves: it keeps CausalMask, LocalMask, and sink-style mask
composition stable enough for cache hits across repeated TPU calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplashMaskCacheKey:
    causal: bool
    local_window: int
    sink_tokens: int
    num_heads: int


def build_splash_mask_cache_key(
    *,
    causal: bool,
    local_window: int,
    sink_tokens: int,
    num_heads: int,
) -> SplashMaskCacheKey:
    return SplashMaskCacheKey(
        causal=causal,
        local_window=max(local_window, 0),
        sink_tokens=max(sink_tokens, 0),
        num_heads=max(num_heads, 1),
    )


def describe_splash_mask_recipe(key: SplashMaskCacheKey) -> dict[str, object]:
    recipe = []
    if key.causal:
        recipe.append("CausalMask")
    if key.local_window > 0:
        recipe.append(f"LocalMask(window={key.local_window})")
    if key.sink_tokens > 0:
        recipe.append(f"SinkMask(tokens={key.sink_tokens})")
    if not recipe:
        recipe.append("FullMask")
    return {
        "cache_key": key,
        "recipe": recipe,
        "message": "Stable mask recipes avoid recompiles from changing Python object identity.",
    }
