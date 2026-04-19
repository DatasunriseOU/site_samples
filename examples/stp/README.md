# STP Examples

This directory holds the MegaCpp POC sample for Semantic Tube Prediction.

What is here:
- `stp_sample.py`: the geodesic auxiliary loss used to penalize unnecessary
  curvature in hidden-state trajectories
- `stp_activation_schedule_sample.py`: the delayed step gate that turns STP on
  after warmup
- `stp_hidden_collection_sample.py`: the single-layer vs multi-layer hidden
  collection contract

What problem it solves:
- it gives the model a cheap regularizer that nudges representations toward
  locally straighter paths
- it does this without requiring a second forward path or a heavy extra head

Where this fits in the model:
- STP is an auxiliary training loss layered onto hidden states after the main
  forward pass
- it is used as a regularizer, not as the primary objective

In simple words:
- one variant uses the last layer only
- one variant uses more random spans to reduce variance
- one variant averages STP across selected intermediate layers
- the step gate lets this regularizer start later in training
