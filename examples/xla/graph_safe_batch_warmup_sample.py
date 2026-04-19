"""Graph-safe runtime warmup scheduling.

What it is: a donor-based excerpt of the helper logic that disables runtime
batch warmup on XLA while preserving scheduled-token accounting helpers.

Why it exists: early-step shape or schedule changes can force recompiles on
compiled backends that expect a stable train-step graph.

What problem it solves: it keeps the TPU runtime graph step-invariant by using
the steady-state accumulation schedule from step 0 on XLA, while retaining the
same token-accounting contract for non-XLA paths.
"""

from __future__ import annotations

import math


def effective_batch_warmup_steps(*, batch_warmup_steps: int, device_type: str) -> int:
    """Normalize runtime batch warmup, disabling it on XLA to keep graphs static."""
    warmup_steps = max(0, int(batch_warmup_steps or 0))
    if device_type == "xla":
        return 0
    return warmup_steps


def resolve_grad_accum_schedule(
    *, total_batch_size: int, world_tokens_per_fwdbwd: int, batch_warmup_steps: int
) -> tuple[int, int]:
    """Return the min/max grad-accum schedule bounds for batch warmup."""
    if total_batch_size % world_tokens_per_fwdbwd != 0:
        raise ValueError(
            "total_batch_size must be divisible by world_tokens_per_fwdbwd"
        )
    max_grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    min_grad_accum_steps = (
        max(1, max_grad_accum_steps // 8)
        if batch_warmup_steps > 0
        else max_grad_accum_steps
    )
    return min_grad_accum_steps, max_grad_accum_steps


def scheduled_grad_accum_steps(
    step: int,
    *,
    batch_warmup_steps: int,
    min_grad_accum_steps: int,
    max_grad_accum_steps: int,
) -> int:
    """Return the grad-accum count for one training step under batch warmup."""
    if batch_warmup_steps <= 0 or step >= batch_warmup_steps:
        return max_grad_accum_steps
    fraction = step / batch_warmup_steps
    return max(
        min_grad_accum_steps,
        round(
            min_grad_accum_steps
            + fraction * (max_grad_accum_steps - min_grad_accum_steps)
        ),
    )


def scheduled_training_tokens(
    num_steps: int,
    *,
    world_tokens_per_fwdbwd: int,
    batch_warmup_steps: int,
    min_grad_accum_steps: int,
    max_grad_accum_steps: int,
) -> int:
    """Return the actual scheduled training tokens over ``num_steps`` steps."""
    steps = max(0, int(num_steps))
    if steps == 0:
        return 0
    warmup_steps = min(steps, max(0, int(batch_warmup_steps or 0)))
    warmup_micro_batches = sum(
        scheduled_grad_accum_steps(
            step,
            batch_warmup_steps=batch_warmup_steps,
            min_grad_accum_steps=min_grad_accum_steps,
            max_grad_accum_steps=max_grad_accum_steps,
        )
        for step in range(warmup_steps)
    )
    full_steps = steps - warmup_steps
    total_micro_batches = warmup_micro_batches + full_steps * max_grad_accum_steps
    return total_micro_batches * world_tokens_per_fwdbwd


def iterations_for_token_target(
    target_tokens: int,
    *,
    world_tokens_per_fwdbwd: int,
    batch_warmup_steps: int,
    min_grad_accum_steps: int,
    max_grad_accum_steps: int,
) -> int:
    """Return the smallest step count whose scheduled tokens reach the target."""
    goal = max(0, math.ceil(float(target_tokens)))
    if goal == 0:
        return 0

    warmup_steps = max(0, int(batch_warmup_steps or 0))
    warmup_tokens = scheduled_training_tokens(
        warmup_steps,
        world_tokens_per_fwdbwd=world_tokens_per_fwdbwd,
        batch_warmup_steps=batch_warmup_steps,
        min_grad_accum_steps=min_grad_accum_steps,
        max_grad_accum_steps=max_grad_accum_steps,
    )
    tokens_per_full_step = world_tokens_per_fwdbwd * max_grad_accum_steps
    if goal > warmup_tokens:
        return warmup_steps + math.ceil((goal - warmup_tokens) / tokens_per_full_step)

    total = 0
    step = 0
    while total < goal:
        total += world_tokens_per_fwdbwd * scheduled_grad_accum_steps(
            step,
            batch_warmup_steps=batch_warmup_steps,
            min_grad_accum_steps=min_grad_accum_steps,
            max_grad_accum_steps=max_grad_accum_steps,
        )
        step += 1
    return step
