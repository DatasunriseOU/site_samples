---
title: "Torch 2.12 TPU/XLA breakage matrix: wheel pain, cache misses, and the workarounds that actually mattered"
description: "A repo-grounded account of where the TPU/XLA stack broke, which failures needed upstream-facing patches, and which ones were better handled as explicit MegaCpp runtime policy."
date: "2026-04-19"
tags: ["PyTorch", "torch-2-12", "xla", "tpu", "wheels", "pjrt"]
summary: >
  Torch 2.12-class TPU/XLA work was less about a clean version upgrade and more
  about surviving a breakage matrix: wheel and ABI drift, broken persistent
  cache reads, missing runtime memory APIs under SPMD, and a handful of launch
  policies that had to become explicit in MegaCpp.
---

# Torch 2.12 TPU/XLA breakage matrix: wheel pain, cache misses, and the workarounds that actually mattered

The practical TPU/XLA story around a Torch 2.12-class stack was not "upgrade the wheel and rerun." It was a breakage matrix. Some failures came from version skew across PyTorch, `torch_xla`, OpenXLA, PJRT, and `libtpu`. Others came from backend behavior that looked acceptable at import time but failed on the real training path.

What mattered in MegaCpp was separating three categories:

- failures that needed a patch against the TPU/XLA stack itself
- failures that were real but better handled as launch or model policy
- features that only existed in newer `torch_xla` lines and therefore had to be gated by version

## The matrix was broader than a version pin

The most useful internal build note was not a single package list but a compatibility bundle. One validated TPU/XLA build recorded PyTorch `2.9.0a0+git21fec65`, `torch_xla 2.9.0+gitc04e61c`, OpenXLA `a76a9a858`, Python 3.13, and a set of local API patches needed just to keep that newer OpenXLA/PJRT layer buildable together. That is the right framing: on TPU/XLA, a version bump is really an ABI and runtime-contract check, not only a packaging event.

The same repo history also documents a later custom stack used for cache testing that moved to PyTorch `2.11.0a0+git7afdbae` with the same `torch_xla` commit family plus local fixes. In other words, even before talking about Torch 2.12 in public, the operational lesson was already clear: a TPU/XLA lane is defined by the whole bundle, not by `torch` alone.

## Breakage matrix

| Surface | Symptom | What MegaCpp did |
| --- | --- | --- |
| persistent compilation cache | cache files were written but restarts still recompiled from scratch | patched `torch_xla` to load serialized executables through the PJRT C API path |
| SPMD memory reporting | in-process HBM reporting failed on virtual `SPMD:0` devices | added a raw runtime-device memory binding and queried physical `TPU:*` devices |
| optimizer graph stability | cache behavior and graph hashes were polluted by scalar extraction in AdamW | forced `capturable=True` for XLA AdamW |
| checkpoint resume on XLA | loading with `assign=True` would replace XLA parameters with CPU tensors | used `assign=False` on XLA resume paths |
| `scan_layers` availability | compile-time improvement feature absent on older `torch_xla` lines | gated the feature and warned unless `torch_xla >= 2.10` |
| `torch.compile` on TPU | compile path could OOM during TPU compilation | kept TPU in eager mode and relied on XLA JIT instead |

That table is the real upgrade artifact. It says which failures were packaging or backend defects and which ones were simply the wrong runtime policy for TPU.

## The most expensive bug was persistent cache that only wrote

The clearest upstream-facing defect was the persistent compilation cache. Internal patch notes document that cache files were successfully written but never read back, so every restart still paid the compile bill. The root cause in the patch write-up is precise: `torch_xla` used `PjRtClient::DeserializeExecutable()`, which returned `UNIMPLEMENTED`, while the working path was `PjRtClient::LoadSerializedExecutable()`, the PJRT C API route that reaches `PJRT_Executable_DeserializeAndLoad` in `libtpu`.

The patch changed two things. It serialized executables first instead of only HLO, and on load it switched from `DeserializeExecutable()` to `LoadSerializedExecutable()`. The documented result was a restart-time improvement from 11.5 seconds to 1.7 seconds on a small validation model, with the larger-model expectation dropping from roughly 47 minutes to roughly 7 minutes.

That distinction matters because the repo also preserves the earlier dead end: HLO-cache writes existed, but they mostly saved tracing overhead rather than the real TPU compile cost. MegaCpp therefore treated executable-cache loading as the actual fix and plain HLO caching as insufficient.

## The SPMD memory problem was an API mismatch, not a dashboard bug

Another failure looked small until it blocked observability. Under XLA SPMD, the Python-visible device could be `SPMD:0`, but memory queries needed the physical runtime devices such as `TPU:18`. The existing binding path rejected those physical strings before they reached the computation client.

The local patch added `_xla_runtime_memory_info(device_str)`, a pybind that bypassed the usual device parsing and forwarded the raw runtime device string directly to the computation client. On the MegaCpp side, the training code was updated to prefer that runtime binding, enumerate physical runtime devices, and fail loudly with an actionable message if the build did not include the patch.

This is a good example of the right public lesson: sometimes a backend feature is not missing, but the Python-visible API is aimed at the wrong abstraction layer. In this case, the correct fix was not a profiler workaround disguised as observability. It was a small binding that exposed the physical-device query path the runtime already understood.

## Some "breakages" were really policy mistakes

Not every TPU failure wanted a framework patch.

MegaCpp ended up treating several issues as launch-policy or model-policy corrections:

- XLA checkpoint resume used `assign=False` so CPU-loaded checkpoint tensors would copy into existing XLA parameters instead of replacing them.
- TPU runs stayed in eager mode because `torch.compile` with the OpenXLA backend could OOM during compilation, while XLA JIT already owned the optimization path.
- XLA AdamW was configured with `capturable=True`, because scalar extraction inside the optimizer step could perturb graph hashing and sabotage cache reuse.

These are important because they change how a breakage matrix should be written. A useful matrix does not label everything as "Torch 2.12 is broken." It distinguishes upstream defects from backend-appropriate policy.

## `scan_layers` was a version and structure gate, not a generic knob

The training stack also documents a practical compile-time optimization via `torch_xla.experimental.scan_layers`, but it is explicitly version-gated and structurally gated. Older `torch_xla` lines did not expose `scan_layers`, so the launch path warns and ignores the flag unless the import succeeds. Even on supported versions, the optimization is only valid when the stacked blocks are structurally homogeneous.

That is the right way to write wheel and version notes. "Feature exists" is too vague. The real statement is: this feature exists only on newer `torch_xla` builds, and it only helps if the model structure obeys the scan contract.

## What Torch 2.12 adds to the story

The GPU-side MegaCpp docs repeatedly record a Torch 2.12 nightly stack as a moving compatibility surface rather than a settled platform. That same habit should be applied to TPU/XLA. The useful question is not "are we on 2.12?" The useful question is which TPU/XLA lane is validated on which full bundle, and which breakages still require local patches or explicit launch policy.

That is why a Torch 2.12 TPU/XLA note should end with a matrix and a stance:

- wheel names alone are not the compatibility story
- cache writes are not proof of cache reuse
- SPMD virtual devices are not the right abstraction for every runtime query
- some failures are backend bugs, but others are just the wrong default policy

## What we would preserve in any future upgrade

If this stack moves again, the MegaCpp habits worth preserving are narrow:

1. Record the full TPU/XLA bundle, not just the `torch` wheel.
2. Separate upstream patch candidates from launch-policy fixes.
3. Keep version-gated features honest about both API availability and structural preconditions.
4. Treat restart-time evidence, not cache-directory existence, as the proof that caching works.
5. Make missing TPU observability bindings fail loudly and specifically.

That is the difference between an install note and an operational compatibility document.

## References

- [TPU bring-up notes](../docs/tpu-bringup-notes.md)
- [TPU backend ownership notes](../docs/tpu-backend-ownership.md)
- [XLA flag profile sample](../examples/xla/xla_flag_profile.py)
- [XLA safe AdamW sample](../examples/xla/xla_safe_adamw.py)
- [Libtpu and JAX ownership notes excerpt](../excerpts/docs/cppmega/xla/libtpu-and-jax-interaction__ownership_notes__v1.md)
