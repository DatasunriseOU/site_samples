---
title: "Mamba3 PsiV cache scaffold"
description: "Why the Mamba3 PsiV cache path is published as a scaffold with a fail-closed gate instead of a silent fallback."
date: "2026-04-19"
tags: ["mamba3", "cache", "scaffold", "runtime"]
---

This is a useful public example because it does not pretend an unfinished cache
path is already a working optimization.

The right contract for a scaffold like this is fail closed. If the gate is
turned on explicitly, the run should refuse to continue until the feature is
implemented. A silent fallback would give the operator the wrong performance
story and make it harder to tell whether the cache path was ever active.

That is why the near-copy example publishes the scaffold state itself. The
interesting thing is not the missing implementation. The interesting thing is
the refusal rule.

## Why this is better than a quiet fallback

Performance scaffolds are dangerous when they lie. A clean refusal is noisy, but
honest. A quiet fallback is easier in the moment and worse for every later
benchmark, profiling run, and regression investigation.

## Example -> article -> upstream docs

- example: [`mamba3_psiv_cache_scaffold.py`](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_psiv_cache_scaffold.py)
- article: [`mamba3-psiv-cache-scaffold`](https://megacpp.com/blog/mamba3-psiv-cache-scaffold/)
- upstream docs: PyTorch environment/no-grad style runtime surfaces and Mamba repository context

## References

- [Mamba3 PsiV cache scaffold example](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/mamba3_psiv_cache_scaffold.py)
- [state-spaces/mamba repository](https://github.com/state-spaces/mamba)
- [PyTorch environment variables docs](https://docs.pytorch.org/docs/stable/torch_environment_variables.html)
