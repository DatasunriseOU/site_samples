#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1;36m### %s\033[0m\n' "$*"; }

say "build"
make -s all

say "reported device attributes"
./query_attrs

say "baseline arch-patch proof"
make -s run-baseline

say "staged tcgen05.alloc gate walk"
make -s probe-alloc-gates

say "optional fuller probes"
echo "  make run-full-alloc"
echo "  make run-full-mma"
echo "  make run-full-tma"
