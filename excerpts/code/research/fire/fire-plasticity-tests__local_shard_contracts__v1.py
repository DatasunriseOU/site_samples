"""Public excerpt.

Source repo: MegaCpp public samples
Purpose: show the local-shard and optimizer-state contracts behind FIRE articles
Edited for clarity.
"""


class FakeDtensor:
    def __init__(self, local_tensor):
        self._local_tensor = local_tensor


def local_tensor_if_dtensor(tensor):
    return getattr(tensor, "_local_tensor", tensor)


def reset_optimizer_states_for_fired_params(optimizer_state, touched):
    for name in touched:
        optimizer_state.pop(name, None)


def test_local_tensor_access_uses_local_view():
    wrapped = FakeDtensor(local_tensor=[[1, 2], [3, 4]])
    assert local_tensor_if_dtensor(wrapped) == [[1, 2], [3, 4]]


def test_optimizer_state_reset_is_selective():
    state = {"w_q": {"exp_avg": 1}, "w_k": {"exp_avg": 2}, "bias": {"exp_avg": 3}}
    reset_optimizer_states_for_fired_params(state, touched={"w_q", "w_k"})
    assert "w_q" not in state
    assert "w_k" not in state
    assert "bias" in state
