# Distributed Memory Notes

This note is the public-safe summary we use when describing distributed C++ model training behavior.

## What usually consumes memory

- model parameters
- optimizer state
- gradients
- activations kept alive across forward and backward
- communication buffers and temporary packing surfaces

## Why a small model can still go out of memory

- activation residency can dominate parameter memory when sequence length and microbatch count rise together
- optimizer state can exceed parameter size by a multiple depending on precision and optimizer design
- pipeline schedules can keep several microbatches resident at once
- expert routing creates bursty token-to-expert imbalance and temporary gather or scatter buffers

## What the split names actually mean

- DP: replicate parameters and split data
- TP: split parameter tensors across ranks
- PP: split layers into pipeline stages
- CP: split sequence-context ownership across ranks
- SP: split sequence work inside tensor-parallel regions
- EP: split expert ownership across ranks

## Public-safe rule

When writing about distributed failures, keep the explanation at the level of tensor ownership, activation lifetime, and scheduling boundaries. Remove workstation paths, hostnames, account names, and cluster identifiers.
