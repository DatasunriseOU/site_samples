"""Public STP sample.

Grounded in local public STP excerpts: the auxiliary term measures how straight
successive hidden-state segments remain, which gives a compact geodesic-style
plasticity signal without article-specific context.
"""


def _difference(vec_a, vec_b):
    return [b - a for a, b in zip(vec_a, vec_b)]


def _dot(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _norm(vec):
    return _dot(vec, vec) ** 0.5


def _cosine(vec_a, vec_b):
    denom = _norm(vec_a) * _norm(vec_b)
    if denom == 0.0:
        return 1.0
    return _dot(vec_a, vec_b) / denom


def stp_loss_sample(hidden_states):
    """Return a minimal geodesic-style curvature penalty for one trajectory."""
    if len(hidden_states) < 3:
        return 0.0

    total = 0.0
    spans = 0
    for idx in range(len(hidden_states) - 2):
        start, middle, end = hidden_states[idx : idx + 3]
        first_leg = _difference(start, middle)
        second_leg = _difference(middle, end)
        total += 1.0 - _cosine(first_leg, second_leg)
        spans += 1
    return total / spans
