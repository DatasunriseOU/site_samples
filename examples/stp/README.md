# STP Examples

This directory holds the MegaCpp POC sample for Semantic Tube Prediction.

What is here:
- `stp_sample.py`: the geodesic auxiliary loss used to penalize unnecessary
  curvature in hidden-state trajectories

What problem it solves:
- it gives the model a cheap regularizer that nudges representations toward
  locally straighter paths
- it does this without requiring a second forward path or a heavy extra head

Where this fits in the model:
- STP is an auxiliary training loss layered onto hidden states after the main
  forward pass
- it is used as a regularizer, not as the primary objective
