# GB10 Repro Bundle

This directory is the GitHub-ready reproduction pack for the public GB10
reverse-engineering story.

It keeps two lanes separate on purpose:

- the public-safe lane that anyone can repeat from the command line with CUDA,
  a patched cubin, and exact Driver API receipts;
- the deeper `driver_patch_lane/` that explores copied-`libcuda` patching and
  helper routing, but is **not** presented as silicon proof.

## What this bundle proves cleanly

- a baseline `sm_100a` cubin can be patched to `sm_121a` by rewriting only the
  ELF `e_flags` architecture field;
- that patched baseline cubin loads, launches, synchronizes, and returns the
  expected arithmetic output on GB10;
- `tcgen05`-oriented probes hit additional software and metadata gates after
  the baseline arch check;
- driver-visible paths and deeper patch lanes are different from a clean
  end-to-end execute receipt.

## What this bundle does not claim

- proven `tcgen05.mma` parity with B200 or GB100;
- proven TMEM availability on GB10;
- that the copied-`libcuda` patch lane is the same thing as clean shipping
  support.

## Start here

1. read [`README_gates.md`](./README_gates.md)
2. read [`README_walkthrough.md`](./README_walkthrough.md)
3. keep [`public_claims.md`](./public_claims.md) nearby while writing about the
   results
4. run [`run.sh`](./run.sh) for the guided public-safe sequence
5. only then look at [`driver_patch_lane/README.md`](./driver_patch_lane/README.md)
   and [`driver_patch_lane/patch_libcuda.py`](./driver_patch_lane/patch_libcuda.py)

## File map

- [`Makefile`](./Makefile): builds the baseline and gate-walk artifacts
- [`run.sh`](./run.sh): one-command public-safe walkthrough
- [`loader.cpp`](./loader.cpp): CUDA Driver API loader with load / launch /
  synchronize receipts
- [`query_attrs.cpp`](./query_attrs.cpp): device-attribute readback for the host
- [`kernel_baseline.cu`](./kernel_baseline.cu): the narrow positive proof lane
- [`kernel_alloc_only.cu`](./kernel_alloc_only.cu): minimal `tcgen05.alloc`
  probe
- [`kernel_sm100a.cu`](./kernel_sm100a.cu): fuller exploratory probe surface
- [`patch_elf.py`](./patch_elf.py): rewrites only the ELF `e_flags` arch field
- [`patch_symbols.py`](./patch_symbols.py): rewrites weak undefined
  reserved-SMEM symbols
- [`patch_nvinfo.py`](./patch_nvinfo.py): strips selected `.nv.info` capability
  records
- [`README_gates.md`](./README_gates.md): compact gate matrix
- [`README_walkthrough.md`](./README_walkthrough.md): exact commands plus
  interpretation
- [`public_claims.md`](./public_claims.md): wording guardrails
- [`driver_patch_lane/README.md`](./driver_patch_lane/README.md): research-only
  explanation of the deeper lane
- [`driver_patch_lane/patch_libcuda.py`](./driver_patch_lane/patch_libcuda.py):
  copied-`libcuda` patch script with public-safe warning header

## Quick start

```bash
make all
./query_attrs
make run-baseline
make probe-alloc-gates
```

For the guided sequence:

```bash
./run.sh
```

## Step-by-step summary

1. Build the baseline and probe cubins with [`Makefile`](./Makefile).
2. Use [`patch_elf.py`](./patch_elf.py) to rewrite `sm_100a -> sm_121a`.
3. Run [`loader.cpp`](./loader.cpp) against the patched baseline cubin.
4. Confirm the positive receipt in [`README_gates.md`](./README_gates.md).
5. Walk the minimal `tcgen05.alloc` lane with [`patch_symbols.py`](./patch_symbols.py)
   and [`patch_nvinfo.py`](./patch_nvinfo.py).
6. Stop the public claim at the point described in
   [`public_claims.md`](./public_claims.md).
7. If you intentionally want the deeper user-space driver lane, move into
   [`driver_patch_lane/`](./driver_patch_lane/).

## Practical scope

The safest way to use this bundle is:

- treat `k_baseline` as the narrow positive proof;
- treat the `k_tcgen05_alloc` walk as evidence of layered software gates;
- treat `kernel_sm100a.cu` as exploratory surface area, not public proof;
- treat [`driver_patch_lane/patch_libcuda.py`](./driver_patch_lane/patch_libcuda.py)
  as research tooling for copied drivers, not a shipping-support claim.

## Related public articles

- `../../articles/gb10-blackwell-tensor-paths-what-we-actually-proved.md`
- `../../articles/gb10-driver-gates-and-false-capability-signals.md`
- `../../articles/gb10-sm100a-cubin-patch-repro.md`
- `../../articles/gb10-libcuda-driver-patch-lane-and-why-it-still-is-not-silicon-proof.md`
