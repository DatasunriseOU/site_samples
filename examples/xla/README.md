# XLA Examples

This directory covers the TPU-side runtime surfaces from the MegaCpp POC.

What is here:
- Pallas-based attention helpers
- Splash or `call_jax` bridges
- sparse-routing helpers for TPU paths
- startup, retry, and calibration helpers
- backend selection receipts
- runtime probes, graph barriers, and fallback policies
- clustered sparse forward-cache helpers for multi-phase TPU sparse attention

What these files are for:
- showing which TPU backend gets picked for which mask or softcap shape
- showing how sparse token routing is normalized before a TPU kernel call
- showing how TPU startup and retry logic is kept deterministic enough for long runs
- showing how runtime probing and XLA graph splitting are kept explicit
- showing how clustered sparse TPU caching keeps static mask semantics separate from per-batch tensors

For this pass, the most relevant files are:
- `xla_backend_dispatch.py`
- `clustered_sparse_jax_interop.py`
- `pallas_softcap_attention_sample.py`
- `xla_call_jax_bridge.py`
- `xla_runtime_probe_sample.py`
- `xla_barrier_schedule_sample.py`
- `xla_backend_fallback_sample.py`
- `splash_mask_cache_sample.py`

In simple words:
- Splash is the stable TPU attention path for plain causal masks.
- Pallas is used when the mask or softcap contract needs custom kernel control.
- chunked sparse helpers are used when the model is not doing dense attention at all.
- runtime probes decide whether TPU is really available or only partially installed.
- graph barriers split giant TPU graphs into smaller compile windows when needed.
