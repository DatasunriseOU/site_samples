# STP Index

This folder currently contains one focused MegaCpp POC sample:

- `stp_sample.py`: the ordered-triplet geodesic loss used for Semantic Tube
  Prediction.
- `stp_activation_schedule_sample.py`: the training-step gate for delayed STP
  activation.
- `stp_hidden_collection_sample.py`: the hidden-state collection rule for
  multi-layer STP.

In simple terms: it samples three positions, compares the two local direction
vectors, and penalizes hidden-state paths that bend more than needed; the other
files show when STP turns on and which hidden states it consumes.
