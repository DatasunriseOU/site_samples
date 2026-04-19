# Core Block Feature Examples

This directory holds public-safe MegaCpp POC examples for feature blocks that sit close to the main model stream.

What is here:
- `engram_branch_sample.py` — a causal n-gram side branch for attention blocks
- `ngram_hash_embedding_sample.py` — token-level hashed local-pattern enrichment at the embedding/input side
- `mhc_branch_mixer_sample.py` — constrained branch mixing for residual, Engram, and other side paths
- `engram_mhc_stack_sample.py` — a small wiring example showing how these features fit together around an A-block

How these features plug into the model:
- n-gram hash runs early and enriches token embeddings before the main block stack
- Engram runs on selected attention blocks and adds a local-pattern branch without creating another full attention layer
- mHC runs after branch construction and mixes active branches back into one stream

What problem they solve:
- n-gram hash gives the model a cheap memory for repeated local token motifs
- Engram helps preserve short syntax and local code fragments in a branch specialized for that job
- mHC prevents multi-branch models from turning into uncontrolled weighted sums

These files are adapted from the MegaCpp POC codebase with only public-safety cleanup applied.
