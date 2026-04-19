"""STP delayed-activation schedule sample.

What it is: a small public-safe receipt for the step gate that turns STP on
only after a configured warmup period.

Why it exists: curvature regularization is intentionally delayed so the base
language-model objective can stabilize first.

What problem it solves: it prevents STP from quietly affecting the earliest
training steps when the model is still forming its basic representations.
"""

from __future__ import annotations


def stp_is_active(*, step: int, start_step: int) -> bool:
    """Mirror the training-step gate used to enable STP."""

    return step >= start_step
