---
title: "Restoration without git history"
description: "How MegaCpp reconstructs a Megatron training tree when the machine still has the code and artifacts but no usable `.git` history."
date: "2026-04-19"
tags: ["megatron", "restoration", "migration", "ops"]
---

The hardest restoration problem is not a clean clone. It is a working machine
image or tarball where the code still exists but the commit graph does not.
Once that happens, the only honest workflow is reconstructive: identify the
likely upstream base, reapply the narrow local patches, and verify the result.

MegaCpp treats that as an explicit operator workflow rather than as folklore.
That is why the restoration recipe and patch README are important public
artifacts. They make the recovery process legible enough to repeat.

## What the workflow preserves

The point is not to recreate every historical commit. The point is to restore a
known-good training tree with enough evidence that later runtime behavior is
explainable. That means preserving:

- the likely upstream base
- the local patch layer
- the verification path after patch application

## References

- [Restoring a Megatron training tree without git history](https://megacpp.com/blog/restoring-a-megatron-training-tree-without-git-history/)
- [NAM56R launch contract sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/nam56r_launch_contract_sample.py)
- [Megatron restoration recipe](https://github.com/DatasunriseOU/cppmega/blob/main/docs/megatron_restoration_recipe.md)
- [Megatron upstream patches README](https://github.com/DatasunriseOU/cppmega/blob/main/cppmega/megatron/upstream_patches/README.md)
