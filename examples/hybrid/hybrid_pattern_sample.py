"""Donor-grounded hybrid pattern helpers for NAM-style layer layouts."""

from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_SYMBOLS = frozenset({"A", "M", "D", "E", "G", "R", "|"})
_TRANSLATED_SYMBOLS = {
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
    requires_custom_mamba: bool
    requires_custom_rnn: bool
    issues: tuple[TranslationIssue, ...]

    @property
    def is_fully_native(self) -> bool:
        return (
            not self.requires_custom_mamba
            and not self.requires_custom_rnn
            and not self.issues
        )


def parse_nem_pattern(pattern: str, depth: int) -> list[str]:
    """Expand a tiled or pipe-delimited NAM pattern to per-layer symbols."""
    if not pattern:
        raise ValueError("pattern must be non-empty")
    upper = pattern.upper()
    invalid = sorted({char for char in upper if char not in _SUPPORTED_SYMBOLS})
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
    return [upper[index % len(upper)] for index in range(depth)]


def count_layer_types(pattern: str, depth: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in parse_nem_pattern(pattern, depth):
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def translate_pattern(
    *,
    pattern: str,
    depth: int,
    force_custom_mamba: bool = True,
) -> HybridPlan:
    """Translate a NAM-style pattern into a public-safe target pattern.

    This preserves the donor semantics that some symbols still require custom
    implementation work and should not be silently treated as native.
    """

    translated: list[str] = []
    issues: list[TranslationIssue] = []
    requires_custom_mamba = False
    requires_custom_rnn = False

    for symbol in parse_nem_pattern(pattern, depth):
        if symbol == "R":
            requires_custom_rnn = True
            issues.append(
                TranslationIssue(
                    symbol="R",
                    message="R has no native translation in this public sample; custom implementation is still required",
                )
            )
            translated.append("R")
            continue

        translated_symbol = _TRANSLATED_SYMBOLS.get(symbol)
        if translated_symbol is None:
            issues.append(
                TranslationIssue(
                    symbol=symbol,
                    message=f"no public translation rule is defined for symbol {symbol!r}",
                )
            )
            translated.append(symbol)
            continue

        if symbol == "M" and force_custom_mamba:
            requires_custom_mamba = True

        translated.append(translated_symbol)

    return HybridPlan(
        source_pattern=pattern,
        translated_pattern="".join(translated),
        requires_custom_mamba=requires_custom_mamba,
        requires_custom_rnn=requires_custom_rnn,
        issues=tuple(issues),
    )
