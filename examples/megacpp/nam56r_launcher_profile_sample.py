"""NAM56R launcher profile sample.

What it is: a public-safe launcher profile that groups the main environment and
parallelism controls used by the real NAM56R training stack.

Why it exists: the real launcher uses many environment variables and flags, and
it is easier to understand them in grouped form than in one long shell file.

What problem it solves: it shows which controls belong to layout, selective
attention, MoE, and runtime-capture policy without exposing any machine-local
paths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LauncherProfile:
    pattern: str
    depth: int
    r_layer_indices: tuple[int, ...]
    dsa_a_layer_ranks: tuple[int, ...]
    tp_size: int
    pp_size: int
    vpp_size: int
    ep_size: int
    micro_batch_size: int
    global_batch_size: int
    mtp_depths: int
    sparse_mode: str
    use_cuda_graphs: bool


DEFAULT_PROFILE = LauncherProfile(
    pattern="AEMEAEMEAEMR",
    depth=52,
    r_layer_indices=(12, 24, 36, 48),
    dsa_a_layer_ranks=(1, 2, 3, 5, 6, 7, 9, 10, 11),
    tp_size=1,
    pp_size=2,
    vpp_size=2,
    ep_size=1,
    micro_batch_size=4,
    global_batch_size=64,
    mtp_depths=2,
    sparse_mode="tilelang",
    use_cuda_graphs=True,
)


def describe_launcher_profile(profile: LauncherProfile = DEFAULT_PROFILE) -> dict[str, object]:
    return {
        "layout": {
            "pattern": profile.pattern,
            "depth": profile.depth,
            "r_layer_indices": list(profile.r_layer_indices),
            "dsa_a_layer_ranks": list(profile.dsa_a_layer_ranks),
        },
        "parallelism": {
            "tp": profile.tp_size,
            "pp": profile.pp_size,
            "vpp": profile.vpp_size,
            "ep": profile.ep_size,
            "micro_batch_size": profile.micro_batch_size,
            "global_batch_size": profile.global_batch_size,
        },
        "runtime": {
            "mtp_depths": profile.mtp_depths,
            "sparse_mode": profile.sparse_mode,
            "use_cuda_graphs": profile.use_cuda_graphs,
        },
    }
