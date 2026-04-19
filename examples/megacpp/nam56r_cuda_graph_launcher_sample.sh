#!/usr/bin/env bash

# CUDA-graph launcher settings for the NAM56R production lane.
#
# What it is: a MegaCpp POC-based excerpt of the production launch flags that scope
# CUDA graph capture for selected model subsystems.
#
# Why it exists: the production lane needed an explicit graph path that could
# be turned on or off per variant without rewriting the whole launch command.
#
# What problem it solves: it makes graph scope, warmup, and the no-graph
# fallback variant explicit, while staying aligned with the actual production
# launcher used for the model family.

set -euo pipefail

variant="${1:-tilelang}"

extra_flags="--recompute-granularity selective --recompute-modules moe_act"
cg_flags="--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess --cuda-graph-warmup-steps 3"

case "${variant}" in
  tilelang)
    echo "[production] tilelang sparse path + CUDA graphs + selective recompute"
    export CPPMEGA_DSA_SPARSE_MODE="tilelang"
    ;;
  gather_scatter)
    echo "[production] gather_scatter sparse path + CUDA graphs + selective recompute"
    export CPPMEGA_DSA_SPARSE_MODE="gather_scatter"
    ;;
  no_cg)
    echo "[production] tilelang sparse path + no CUDA graphs + selective recompute"
    export CPPMEGA_DSA_SPARSE_MODE="tilelang"
    cg_flags=""
    ;;
  *)
    echo "unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

printf 'EXTRA_FLAGS=%s\n' "$extra_flags"
printf 'CG_FLAGS=%s\n' "$cg_flags"
