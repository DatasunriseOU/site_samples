# XLA Examples

This directory covers the TPU-side runtime surfaces from the MegaCpp POC.

What is here:
- Pallas-based attention helpers
- Splash or `call_jax` bridges
- sparse-routing helpers for TPU paths
- startup, retry, and calibration helpers
- backend selection receipts

What these files are for:
- showing which TPU backend gets picked for which mask or softcap shape
- showing how sparse token routing is normalized before a TPU kernel call
- showing how TPU startup and retry logic is kept deterministic enough for long runs

For this pass, the most relevant files are:
- `xla_backend_dispatch.py`
- `clustered_sparse_jax_interop.py`
- `pallas_softcap_attention_sample.py`
- `xla_call_jax_bridge.py`

In simple words:
- Splash is the stable TPU attention path for plain causal masks.
- Pallas is used when the mask or softcap contract needs custom kernel control.
- chunked sparse helpers are used when the model is not doing dense attention at all.

