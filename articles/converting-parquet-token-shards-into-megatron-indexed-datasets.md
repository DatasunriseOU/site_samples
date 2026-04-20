---
title: "Converting parquet token shards into Megatron indexed datasets"
description: "Why MegaCpp keeps a narrow data bridge from tokenized parquet shards to Megatron indexed datasets instead of tying data preparation to one runtime import surface."
date: "2026-04-19"
tags: ["data", "megatron", "parquet", "dataset", "migration"]
---

The interesting data bridge is not tokenization by itself. It is the handoff
between a tokenized columnar corpus and the indexed dataset format the training
runtime actually expects.

MegaCpp keeps that bridge explicit. The converter and thinner format wrapper
exist so the dataset contract stays readable even when the full Megatron import
surface is not available in the environment doing the conversion.

## Why this bridge deserves its own public example

It solves a very specific operational problem. Data preparation often happens on
machines or in environments that are not identical to the final training lane.
If the conversion step is too tightly coupled to one training runtime import
surface, the pipeline becomes harder to port and harder to validate.

The public examples keep the bridge narrow instead:

- one sample for parquet-to-indexed conversion
- one sample for naming and split formatting

That is enough to describe the contract without pretending the whole training
tree has to be present on the data-prep machine.

## References

- [Parquet to Megatron indexed dataset sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/parquet_to_megatron_indexed_dataset_sample.py)
- [Prepare-format MegaCpp sample](https://github.com/DatasunriseOU/site_samples/blob/main/examples/megacpp/prepare_format_megacpp_sample.py)
- [Packed rows as the real training contract](https://megacpp.com/blog/packed-rows-as-the-real-training-contract/)
- [Data preparation docs](https://github.com/DatasunriseOU/cppmega/blob/main/docs/data_preparation.md)
