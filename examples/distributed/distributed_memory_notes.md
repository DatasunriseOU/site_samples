# Distributed Memory Notes

This note is a public-safe summary derived from donor code in `cpu_offload.py`, `meta_init.py`, `memory_estimator.py`, and `memory_debug.py`.

## What usually consumes memory

- parameters
- gradients
- optimizer state
- activations
- MoE routing and other feature-specific activations
- runtime-reserved allocator space and framework overhead

## Why a small model can still go out of memory

- activation residency can dominate parameter memory once sequence length, microbatch count, or checkpoint boundaries keep more tensors alive at once
- optimizer state can exceed parameter memory by a multiple depending on optimizer choice and precision
- pipeline and accumulation schedules can keep several microbatches resident simultaneously
- MoE routing can add bursty dispatch, gather, and imbalance buffers that are not obvious from parameter counts alone
- allocator reservation and fragmentation can make `reserved` memory materially higher than `allocated` memory

## Donor-backed triage order

- Start with a bucketed estimate: parameters, gradients, optimizer, activations, MoE routing, feature activations, runtime reserve, and fixed overhead
- If runtime still fails, inspect allocator counters such as current and peak allocated bytes, current and peak reserved bytes, allocation retries, OOM count, and inactive split bytes
- Separate model-state memory from residual runtime memory by subtracting parameter, gradient, and buffer bytes from total allocated bytes
- Only then choose the mitigation: reduce activation lifetime, narrow offload targets, lower optimizer footprint, or trim routing and temporary buffers

## What selective activation offload really means

- The donor offload path is opt-in and module-targeted, not a blanket "offload the whole model" switch
- Valid target families include attention projections, the full attention module, dense or MoE feed-forward blocks, and selected auxiliary branches
- Large saved tensors are the intended candidates; small bookkeeping tensors stay on device

## What meta-init actually protects

- Instantiate the model on the meta device so tensors have metadata only
- Materialize empty tensors directly on the target device with `to_empty(device=...)`
- Run initialization after materialization so no parameter remains on the meta device and no full-size intermediate CPU copy is required

## What the split names actually mean

- DP: replicate parameters and split data
- TP: split parameter tensors across ranks
- PP: split layers into pipeline stages
- CP: split sequence-context ownership across ranks
- SP: split sequence work inside tensor-parallel regions
- EP: split expert ownership across ranks

## Public-safe rule

When writing about distributed failures, keep the explanation at the level of tensor ownership, activation lifetime, allocator behavior, and scheduling boundaries. Remove workstation paths, hostnames, account names, cluster identifiers, and any internal run labels.
