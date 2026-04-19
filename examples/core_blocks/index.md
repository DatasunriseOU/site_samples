# Core Blocks Index

Use this folder when you want to understand the non-attention side features that were attached directly to the MegaCpp model blocks.

Quick map:
- Input enrichment: `ngram_hash_embedding_sample.py`
- Attention-side local branch: `engram_branch_sample.py`
- Branch reconciliation: `mhc_branch_mixer_sample.py`
- End-to-end local contract: `engram_mhc_stack_sample.py`
- Token-wise depth control: `gateskip_residual_router_sample.py`, `flexidepth_adapter_sample.py`
- Block family map: `block_taxonomy_sample.py`
- Residual alternatives: `residual_paths_sample.py`

Simple mental model:
1. Tokens can receive local-pattern hints through the n-gram hash module.
2. Selected attention blocks can build an Engram branch from nearby context.
3. mHC mixes the block outputs so the residual stream stays stable instead of becoming a hand-tuned branch soup.
4. GateSkip and FlexiDepth are the lightweight ways to skip work while keeping shapes static.
5. A/M/E/C/R block names tell you where each feature family actually lives.

These are examples, not full training modules. The goal is to show the real feature contracts and why they exist.
