# XLA backend index

## What this directory covers

These examples document the TPU backend families behind MegaCpp training and
runtime execution.

## Backend families

### Native Pallas via `trace_pallas`

Use this lane when the TPU attention kernel needs fused softcap, local-window
masking, or document-boundary masking without falling back to a slower bridge.

Relevant examples:

- `pallas_softcap_attention_sample.py`
- `trace_pallas_scalar_prefetch_sample.py`
- `pallas_grid_shrinking_sample.py`
- `xla_pallas_bridge_receipt.py`

### Splash and Splash Local compatibility helpers

Use this lane when the runtime needs Splash-style sparse-local contracts,
bridge validation, or JAX-side helper interoperability.

Relevant examples:

- `xla_call_jax_bridge.py`
- `call_jax_bridge_runtime.py`
- `splash_local_mask_builder_sample.py`
- `xla_backend_dispatch.py`

### Segment-ID and document masking support

These samples show how TPU runs keep document boundaries without constructing a
dense token-by-token mask.

Relevant examples:

- `xla_segment_ids_layout_sample.py`
- `pallas_softcap_attention_sample.py`
- `xla_call_jax_bridge.py`

### Clustered sparse and validity helpers

These files cover the narrower experimental sparse TPU lane where validity,
mask shape, and interop signatures must stay aligned with the kernel surface.

Relevant examples:

- `clustered_sparse_jax_interop.py`
- `graph_safe_batch_warmup_sample.py`

## Model/runtime placement in simple words

- Attention chooses one TPU backend family depending on the mask type,
  softcap, and whether the native path is available.
- Segment IDs tell the kernel which tokens belong to the same document.
- Splash Local limits each token to a nearby window so very long contexts do
  not require a dense quadratic mask.
- The clustered sparse path is the experimental lane for stricter sparse
  routing geometry on TPU.
