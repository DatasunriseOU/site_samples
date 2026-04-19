# Hybrid Examples

This directory shows how the MegaCpp POC combines different sequence-mixing and
memory branches inside one model family.

These files are here to answer simple practical questions:
- Which attention-like paths can live inside an A-block?
- Which sparse-attention backend knobs actually change runtime behavior?
- How do mHC and residual-related options interact?
- How does MoD routing decide whether a token uses the full block or a skip path?
- What adapter and LoRA metadata keeps runtime composition safe?
- Where do DeltaNet and hyper-connections sit in the block menu?
- What do short names like A-block, M-block, D-block, E-block, and R-block mean?

Files in this directory:
- `hybrid_pattern_sample.py`: grounded summary of the unified A-block and superblock wiring.
- `attention_backend_variants_sample.py`: the real sparse-attention backend and helper-choice surface.
- `mhc_stream_residual_sample.py`: the real config knobs around mHC, FP32 residual, and AttnRes conflicts.
- `mod_routing_surface_sample.py`: the real MoD routing modes and their gather-vs-gate tradeoffs.
- `modr_recurrent_lora_wiring_sample.py`: the recurrent shared-core + LoRA-branch wiring contract used by MoDr.
- `modr_router_bookkeeping_sample.py`: the branch-routing bookkeeping and optional auxiliary-loss contract used by MoDr.
- `adapter_lora_runtime_sample.py`: the runtime adapter metadata contract for LoRA, DoRA, VeRA, QLoRA, and DyLoRA.
- `deltanet_hyperconnection_sample.py`: the block-selection receipt for DeltaNet and multi-stream hyper-connections.
- `block_taxonomy_sample.py`: simple public taxonomy for A/M/D/E/R/C block labels.

How this plugs into the model:
- The A-block constructor decides whether dense attention, sparse attention, and Engram are present.
- The superblock constructor decides how many hidden-state streams exist and whether hyper-connections are static or dynamic.
- GPT-level config controls whether mHC owns cross-layer stream mixing and whether other residual helpers can stay active.
- MoD config decides whether tokens are top-k routed, threshold routed, or softly gated without gather/scatter.
- MoDr keeps one recurrent shared block and reuses small LoRA branches to explore different reasoning paths without cloning the whole block.
- The adapter stack preserves enough LoRA metadata for safe runtime composition and online adaptation.

In simple words:
- Dense attention is the normal full-context path.
- Sparse attention is the cheaper selective path.
- Engram is an extra learned memory branch.
- mHC mixes several hidden-state streams instead of carrying only one stream forward.
- MoD decides which tokens pay for full block compute.
- DeltaNet is a recurrent sequence-mixer option selected by the same hybrid pattern family.
