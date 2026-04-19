"""Public-safe optimizer-state reset companion for FIRE."""


def reset_optimizer_states_for_fired_params(optimizer_state: dict[str, object], fired_params: list[str]) -> dict[str, object]:
    remaining = dict(optimizer_state)
    for name in fired_params:
        remaining.pop(name, None)
    return remaining
