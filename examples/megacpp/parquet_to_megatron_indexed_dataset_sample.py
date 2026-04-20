"""MegaCpp public example: convert parquet token shards into Megatron indexed data.

What this solves in simple words:
- training data may already exist as tokenized parquet shards;
- Megatron training wants indexed `.bin/.idx` style datasets;
- a converter should preserve the token contract even when the runtime import
  surface is not fully available.
"""

from __future__ import annotations


def build_conversion_contract() -> dict[str, object]:
    return {
        "input": "tokenized parquet shards",
        "output": "MMapIndexedDataset-style bin/idx",
        "fallback_writer_allowed": True,
        "why_it_exists": "keep the data bridge usable even when Megatron imports are unavailable",
    }
