"""Near-copy MegaCpp POC example: author-pure Mamba3 spec seam.

This sample keeps the important contract from the real integration visible:
when the Megatron-side Mamba layer uses an identity norm surface, the author
Mamba3 path still needs an explicit RMSNorm before projection or the residual
stream drifts immediately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuthorMamba3Seam:
    uses_identity_norm: bool
    explicit_pre_norm: bool
    tensor_parallel_supported: bool
    context_parallel_supported: bool
    packed_sequences_supported: bool
    inference_supported: bool
    notes: str


def author_mamba3_training_path() -> AuthorMamba3Seam:
    return AuthorMamba3Seam(
        uses_identity_norm=True,
        explicit_pre_norm=True,
        tensor_parallel_supported=False,
        context_parallel_supported=False,
        packed_sequences_supported=False,
        inference_supported=False,
        notes="RMSNorm is applied explicitly before in_proj to compensate for the missing fused norm",
    )


def author_mamba3_without_pre_norm() -> AuthorMamba3Seam:
    return AuthorMamba3Seam(
        uses_identity_norm=True,
        explicit_pre_norm=False,
        tensor_parallel_supported=False,
        context_parallel_supported=False,
        packed_sequences_supported=False,
        inference_supported=False,
        notes="residual magnitudes can grow unbounded through the stack",
    )


def compare_author_mamba3_paths() -> dict[str, AuthorMamba3Seam]:
    return {
        "with_explicit_pre_norm": author_mamba3_training_path(),
        "without_explicit_pre_norm": author_mamba3_without_pre_norm(),
    }
