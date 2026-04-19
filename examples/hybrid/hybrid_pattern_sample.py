"""This example translates a NAM-style hybrid layer pattern into a public-safe Megatron pattern.

Why it exists: the model mixes attention, Mamba, and routed layers in one
depth schedule, so launch code needs a deterministic translation step.

What problem it solves: it makes the pattern explicit and fails closed on
symbols that still need custom runtime support instead of silently remapping
them.
"""

from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_SYMBOLS = frozenset({"A", "M", "D", "E", "G", "R", "|"})
_NEMOTRON_TO_MEGATRON = {
    "A": "*",
    "M": "M",
    "D": "G",
    "E": "E",
}


@dataclass(frozen=True)
class TranslationIssue:
    symbol: str
    message: str


@dataclass(frozen=True)
class HybridPlan:
    source_pattern: str
    translated_pattern: str
    requires_custom_mamba3: bool
    requires_custom_m2rnn: bool
    requires_mtp_suffix: bool
    issues: tuple[TranslationIssue, ...]

    @property
    def is_fully_native(self) -> bool:
        return (
            not self.requires_custom_mamba3
            and not self.requires_custom_m2rnn
            and not self.issues
        )


def parse_nem_pattern(pattern: str, depth: int) -> list[str]:
    """Match the MegaCpp POC tiling rules for NAM-style layer patterns."""

    if not pattern:
        raise ValueError("pattern must be non-empty")
    upper = pattern.upper()
    invalid = sorted({ch for ch in upper if ch not in _SUPPORTED_SYMBOLS})
    if invalid:
        raise ValueError(
            f"invalid pattern chars {invalid!r}; supported symbols are A, M, D, E, G, R and |"
        )

    if "|" in upper:
        segments = upper.split("|")
        flat = "".join(segments)
        if len(flat) != depth:
            raise ValueError(
                f"pipe-delimited pattern expands to {len(flat)} layers, expected depth={depth}"
            )
        return list(flat)

    return [upper[i % len(upper)] for i in range(depth)]


def count_layer_types(pattern: str, depth: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in parse_nem_pattern(pattern, depth):
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def translate_pattern(
    *,
    pattern: str,
    depth: int,
    mtp_depths: int = 0,
    force_author_mamba3: bool = True,
) -> HybridPlan:
    """Translate a NAM-style pattern into Megatron hybrid syntax.

    The important MegaCpp POC contract is that `R` does not get silently remapped,
    and `M` can still require a custom Mamba3-backed runtime even when the
    textual Megatron symbol is available.
    """

    translated: list[str] = []
    issues: list[TranslationIssue] = []
    requires_custom_m2rnn = False
    requires_custom_mamba3 = False

    for symbol in parse_nem_pattern(pattern, depth):
        if symbol == "R":
            requires_custom_m2rnn = True
            issues.append(
                TranslationIssue(
                    symbol="R",
                    message=(
                        "R has no native Megatron equivalent in this public sample; "
                        "custom runtime support is still required"
                    ),
                )
            )
            translated.append("R")
            continue

        translated_symbol = _NEMOTRON_TO_MEGATRON.get(symbol)
        if translated_symbol is None:
            issues.append(
                TranslationIssue(
                    symbol=symbol,
                    message=f"no translation rule is defined for symbol {symbol!r}",
                )
            )
            translated.append(symbol)
            continue

        if symbol == "M" and force_author_mamba3:
            requires_custom_mamba3 = True

        translated.append(translated_symbol)

    translated_pattern = "".join(translated)
    requires_mtp_suffix = mtp_depths > 0
    if requires_mtp_suffix:
        translated_pattern = translated_pattern + "/" + "/".join("*-" for _ in range(mtp_depths))

    return HybridPlan(
        source_pattern=pattern,
        translated_pattern=translated_pattern,
        requires_custom_mamba3=requires_custom_mamba3,
        requires_custom_m2rnn=requires_custom_m2rnn,
        requires_mtp_suffix=requires_mtp_suffix,
        issues=tuple(issues),
    )
