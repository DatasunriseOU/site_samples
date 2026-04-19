# STP Index

This folder currently contains one focused MegaCpp POC sample:

- `stp_sample.py`: the ordered-triplet geodesic loss used for Semantic Tube
  Prediction.

In simple terms: it samples three positions, compares the two local direction
vectors, and penalizes hidden-state paths that bend more than needed.
