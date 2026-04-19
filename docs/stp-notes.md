# STP Notes

This note summarizes the public trajectory-straightness auxiliary-loss
story used in MegaCpp articles.

## Core loss shape

The public sample keeps the loss intentionally narrow.

- It consumes existing hidden states.
- It samples ordered triples of positions.
- It forms two local direction vectors.
- It penalizes curvature with `1 - cosine_similarity`.

This supports the public claim that the feature is a cheap local-linearity
regularizer rather than a second prediction tower.

## Variants the public sample supports

- A last-layer single-span variant.
- A last-layer multi-span variant.
- A multi-layer variant that averages the same loss across an explicit list of
  hidden-state tensors.

The important policy point is that layer choice is explicit. The loss kernel
does not silently decide which layers to supervise.

## Runtime policy

The public story should describe the start-step gate as a schedule policy,
not as a universal STP fact. A delayed start simply means the auxiliary term is
enabled after the base run has entered a more stable regime.

That also means STP receipts should be interpreted only over intervals where the
feature is actually live.

## Backend posture

The public sample uses fixed-rank tensor operations with no host-side rejection
loop and no variable-size outputs. That is the right support surface for
describing the implementation as graph-friendly and easy to transport across
backends.

## References

- https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/code/research/stp/stp-geodesic-regularizer__stp_loss_surface__v1.py
- https://github.com/DatasunriseOU/site_samples/blob/main/excerpts/docs/research/stp/stp-after-ten-thousand-steps__activation_gate_note__v1.md
- https://arxiv.org/abs/2602.22617
