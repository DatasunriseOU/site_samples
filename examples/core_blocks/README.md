# Core Block Feature Examples

This directory holds public-safe MegaCpp POC examples for feature blocks that sit close to the main model stream.

What is here:
- `engram_branch_sample.py` — a causal n-gram side branch for attention blocks
- `ngram_hash_embedding_sample.py` — token-level hashed local-pattern enrichment at the embedding/input side
- `mhc_branch_mixer_sample.py` — constrained branch mixing for residual, Engram, and other side paths
- `engram_mhc_stack_sample.py` — a small wiring example showing how these features fit together around an A-block
- `gateskip_residual_router_sample.py` — residual gating for token-wise layer skipping
- `gateskip_loss_bookkeeping_sample.py` — GateSkip sparsity-loss and budget bookkeeping
- `flexidepth_adapter_sample.py` — static-shape layer skipping with a router and lightweight adapter
- `flexidepth_loss_stats_sample.py` — FlexiDepth skip loss and layer-usage stats
- `flexidepth_frozen_adapter_wiring_sample.py` — frozen-backbone adapter wiring for FlexiDepth
- `block_taxonomy_sample.py` — A/M/E/C/R block family map
- `residual_paths_sample.py` — fp32 residual, AttnRes, and mHC interaction summary

How these features plug into the model:
- n-gram hash runs early and enriches token embeddings before the main block stack
- Engram runs on selected attention blocks and adds a local-pattern branch without creating another full attention layer
- mHC runs after branch construction and mixes active branches back into one stream

What problem they solve:
- n-gram hash gives the model a cheap memory for repeated local token motifs
- Engram helps preserve short syntax and local code fragments in a branch specialized for that job
- mHC prevents multi-branch models from turning into uncontrolled weighted sums
- GateSkip and FlexiDepth reduce unnecessary block compute without changing sequence shape
- GateSkip bookkeeping keeps sparsity pressure and token-budget scheduling tied to the actual gate tensors
- FlexiDepth bookkeeping tracks how many layers tokens really use and how the skip loss mixes with LM loss
- FlexiDepth frozen wiring shows how the router and adapter can move while the base block stays fixed
- the block taxonomy clarifies which block family is responsible for which capability
- the residual path summary shows which branch-mixing alternatives can conflict

These files are adapted from the MegaCpp POC codebase with only public-safety cleanup applied.
