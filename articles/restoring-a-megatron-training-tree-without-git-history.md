---
title: "Restoring a Megatron training tree without git history"
description: "How MegaCpp treats restoration as a patch-and-verification workflow when the machine backup has the code but not the `.git` metadata."
date: "2026-04-19"
tags: ["megatron", "migration", "restoration", "ops"]
---

The interesting restoration problem is not a fresh clone. It is the machine
that still has the working tree, the data, and the receipts, but no `.git`
history. In that situation, the only honest workflow is reconstructive:
identify the likely upstream base, reapply the narrow local patches, and verify
the resulting tree directly.

That is why MegaCpp's restoration docs matter. They treat restoration as a
repeatable operator workflow rather than as archaeology by guesswork.

## References

- [Megatron restoration recipe](https://github.com/DatasunriseOU/cppmega/blob/main/docs/megatron_restoration_recipe.md)
- [Megatron upstream patches README](https://github.com/DatasunriseOU/cppmega/blob/main/cppmega/megatron/upstream_patches/README.md)
- [Migration matrix](https://github.com/DatasunriseOU/cppmega/blob/main/docs/migration_matrix.md)
