"""Plasticity phase scheduler sample.

What it is: a small public-safe scheduler that coordinates when FIRE, DASH,
and ReDo should run during one training program.

Why it exists: the three plasticity tools solve different problems and are not
meant to fire on the same cadence.

What problem it solves: it keeps one-time phase surgery, periodic shrinking,
and periodic dormant-neuron repair from collapsing into one vague "reset"
button.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlasticitySchedule:
    fire_between_phases: bool
    dash_every_steps: int
    redo_every_steps: int
    context_extension_start_step: int


def plan_plasticity_actions(
    *,
    step: int,
    in_context_extension: bool,
    schedule: PlasticitySchedule,
) -> dict[str, bool]:
    """Return which plasticity tools should run at a given step."""

    run_fire = schedule.fire_between_phases and in_context_extension and step == schedule.context_extension_start_step
    run_dash = schedule.dash_every_steps > 0 and step > 0 and step % schedule.dash_every_steps == 0
    run_redo = schedule.redo_every_steps > 0 and step > 0 and step % schedule.redo_every_steps == 0
    return {
        "run_fire": run_fire,
        "run_dash": run_dash,
        "run_redo": run_redo,
    }
