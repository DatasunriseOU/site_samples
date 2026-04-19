# TPU Backend Ownership

This note records the public rules MegaCpp uses when talking about TPU
execution.

## A TPU run has several owners

| Layer | Typical responsibility |
| --- | --- |
| application config | backend selection, fallback policy, logging |
| framework frontend | PyTorch/XLA or JAX tracing and execution model |
| PJRT runtime | runtime interface between frontend and accelerator backend |
| TPU runtime stack | device-specific execution and version compatibility |

## Operational rules

- explicit backend choice beats heuristic autodetect
- unsupported feature combinations should fail closed
- fallback must be visible in logs and receipts
- provenance should record what backend actually executed
- compile, runtime, and data-pipeline issues should be debugged separately

## Public takeaway

Most TPU failures are not "the TPU does not exist" failures. They are ownership
failures: the wrong backend was selected, a fallback hid the real problem, or a
frontend/runtime mismatch was interpreted as a model bug.
