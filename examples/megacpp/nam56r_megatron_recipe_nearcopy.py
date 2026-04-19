"""Near-copy MegaCpp POC example: NAM56R Megatron translation contract.

This sample keeps the real public rule visible: NAM-style layer patterns do not
map 1:1 into Megatron-native syntax, so the translation plan has to fail closed
on unsupported symbols and report which features still need custom seams.
"""

from __future__ import annotations

from dataclasses import dataclass


NEMOTRON_TO_MEGATRON = {
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
class MegatronHybridPlan:
    source_pattern: str
    translated_pattern: str
    requires_custom_mamba3: bool
    requires_custom_m2rnn: bool
    requires_mtp_suffix: bool
    issues: tuple[TranslationIssue, ...]


def parse_pattern(pattern: str, depth: int) -> list[str]:
    if not pattern:
        raise ValueError("pattern must be non-empty")
    upper = pattern.upper()
    if "|" in upper:
        flat = "".join(upper.split("|"))
        if len(flat) != depth:
            raise ValueError("pipe-delimited pattern does not match depth")
        return list(flat)
    return [upper[i % len(upper)] for i in range(depth)]


def translate_pattern(pattern: str, depth: int, mtp_depths: int = 0) -> MegatronHybridPlan:
    translated: list[str] = []
    issues: list[TranslationIssue] = []
    requires_custom_mamba3 = False
    requires_custom_m2rnn = False

    for symbol in parse_pattern(pattern, depth):
        if symbol == "R":
            requires_custom_m2rnn = True
            translated.append("R")
            issues.append(
                TranslationIssue(
                    symbol="R",
                    message="M2RNN has no Megatron-native equivalent and must remain a custom seam",
                )
            )
            continue

        mapped = NEMOTRON_TO_MEGATRON.get(symbol)
        if mapped is None:
            translated.append(symbol)
            issues.append(
                TranslationIssue(symbol=symbol, message="no Megatron translation rule defined")
            )
            continue

        if symbol == "M":
            requires_custom_mamba3 = True
        translated.append(mapped)

    translated_pattern = "".join(translated)
    requires_mtp_suffix = mtp_depths > 0
    if requires_mtp_suffix:
        translated_pattern += "/" + "/".join("*-" for _ in range(mtp_depths))

    return MegatronHybridPlan(
        source_pattern=pattern,
        translated_pattern=translated_pattern,
        requires_custom_mamba3=requires_custom_mamba3,
        requires_custom_m2rnn=requires_custom_m2rnn,
        requires_mtp_suffix=requires_mtp_suffix,
        issues=tuple(issues),
    )


def build_nam56r_reference_plan() -> MegatronHybridPlan:
    return translate_pattern(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1)
