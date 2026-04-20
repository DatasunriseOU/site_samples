---
title: "DSA indexer memory fix"
description: "Why MegaCpp replaces a memory-hungry DSA score path with a fused top-k scoring surface and treats that change as a systems fix, not just a kernel tweak."
date: "2026-04-19"
tags: ["dsa", "memory", "attention", "H200"]
---

Some attention fixes are really memory fixes in disguise. The DSA indexer path
is one of them. If the score path materializes the wrong intermediate, the
runtime spends memory on a tensor the later top-k logic did not actually need.

The public sample keeps the right lesson visible: fused top-k scoring is not
only about speed. It is about removing an avoidable memory bill from the hot
path.

## References

- [DSA indexer memory sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_indexer_memory_sample.py)
- [DSA CUDA graph safety sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/dsa_cuda_graph_safety_sample.py)
- [Expert parallel and MoE sharding](https://megacpp.com/blog/expert-parallel-and-moe-sharding/)
