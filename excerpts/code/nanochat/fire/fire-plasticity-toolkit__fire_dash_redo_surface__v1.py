"""Sanitized public excerpt.

Source repo: MegaCpp research repo
Source file: plasticity toolkit module
Purpose: show the combined FIRE, DASH, and ReDo control surface
Edited for public clarity.
"""

def apply_plasticity_toolkit(model, optimizer, step, cfg):
    if cfg.fire_enabled and step in cfg.fire_steps:
        apply_fire(model, mode=cfg.fire_mode)
        reset_optimizer_states_for_fired_params(optimizer)

    if cfg.dash_enabled and step % cfg.dash_interval == 0:
        dash_step(model, alpha=cfg.dash_alpha)

    if cfg.redo_enabled and step % cfg.redo_interval == 0:
        redo_dormant_neurons(model, threshold=cfg.redo_threshold)
