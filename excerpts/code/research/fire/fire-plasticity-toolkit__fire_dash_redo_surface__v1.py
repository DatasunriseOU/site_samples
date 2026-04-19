"""Public excerpt.

Source repo: MegaCpp public samples
Reference note: docs/plasticity-toolkit-notes.md
Purpose: show the combined FIRE, DASH, and ReDo control surface
Edited for clarity.
"""

def _is_dtensor_like(tensor) -> bool:
    return hasattr(tensor, "_local_tensor")


def _local_tensor_if_dtensor(tensor):
    return getattr(tensor, "_local_tensor", tensor)


def _shard0_bounds(rows: int, world_size: int, rank: int) -> tuple[int, int]:
    base, remainder = divmod(rows, world_size)
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    return start, stop


def _match_grad_to_local_shard(grad_rows, local_rows: int):
    return grad_rows[:local_rows]


def _is_xla_tensor(tensor) -> bool:
    return str(getattr(tensor, "device", "")).startswith("xla")


def apply_fire(model, mode: str):
    return {"attn_q", "attn_k", "mlp_up"} if mode else set()


def reset_optimizer_states_for_fired_params(optimizer, touched):
    optimizer["reset_for"] = sorted(touched)


def dash_step(model, alpha: float):
    return {"alpha": alpha, "updated": True}


def redo_dormant_neurons(model, threshold: float):
    return {"threshold": threshold, "recycled": True}


def apply_plasticity_toolkit(model, optimizer, step, cfg):
    if cfg.fire_enabled and step in cfg.fire_steps and not _is_xla_tensor(model):
        touched = apply_fire(model, mode=cfg.fire_mode)
        reset_optimizer_states_for_fired_params(optimizer, touched)

    if cfg.dash_enabled and step % cfg.dash_interval == 0:
        dash_step(model, alpha=cfg.dash_alpha)

    if cfg.redo_enabled and step % cfg.redo_interval == 0:
        redo_dormant_neurons(model, threshold=cfg.redo_threshold)
