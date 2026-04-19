---
title: "DSA index-cache patch"
description: "Why caching sparse top-k indices across selected DSA layers is not just a speed trick, and why the shared path has to fail closed back to a full layer when no valid cache is available."
date: "2026-04-19"
tags: ["dsa", "cache", "sparse-attention", "patches"]
---

This patch is interesting because it looks like a local optimization, but the
real contract is larger.

Some DSA layers compute sparse indices and later layers reuse them. That only
works if the reuse path is explicit and the failure mode is safe. A shared
layer without a valid preceding cache cannot keep pretending it is on the cheap
path. It has to promote itself back to a full path and recompute.

That is the public rule the near-copy sample preserves: cache when the contract
exists, fail closed when it does not.

## Why this matters beyond one patch

Sparse-attention caches are tempting to describe as obvious wins. They are not
obvious if the lifecycle is underspecified. Cache invalidation, nearest valid
source, and cross-stage absence all change whether reuse is safe.

That is why this public example is worth keeping. It documents the schedule and
the fallback rule instead of implying cached sparse indices are globally valid.

## Example -> article -> upstream docs

- example: [`index_cache_patch_nearcopy.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/index_cache_patch_nearcopy.py)
- article: [`dsa-index-cache-patch`](https://megacpp.com/blog/dsa-index-cache-patch/)
- upstream docs: Megatron-LM repository context and sparse-attention implementation references

## References

- [DSA index-cache patch near-copy example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/index_cache_patch_nearcopy.py)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch no_grad docs](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
