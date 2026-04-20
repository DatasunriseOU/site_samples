---
title: "Reproducing the sm_100a -> sm_121a Cubin Patch on GB10: CUDA/C++ Code, ELF Edits, and the Exact Point Where tcgen05 Stops"
description: "A practical GB10 reproduction guide for the narrow result we can defend publicly: a patched sm_100a baseline cubin executes on GB10, while tcgen05-oriented probes stop at later driver-side gates rather than producing a publication-grade tcgen05 proof."
date: "2026-04-20"
tags: ["GB10", "Blackwell", "CUDA", "C++", "cubin", "tcgen05", "driver-research"]
---

This article is the step-by-step version of the conservative GB10 claim.

What we can defend publicly is narrow:

- a simple `sm_100a` cubin can be patched so it loads and executes on GB10;
- the same workflow becomes a layered gate walk once the kernel uses `tcgen05`-oriented instructions;
- the clean public evidence still stops short of proving working `tcgen05.mma` parity on GB10.

That is enough to be useful. It tells you what to reproduce, which patches matter, which commands to run, and exactly where the public-safe story stops.

## What you need

The public repro bundle for this article lives here:

- [README.md](../examples/megacpp/gb10_repro_bundle/README.md)
- [Makefile](../examples/megacpp/gb10_repro_bundle/Makefile)
- [run.sh](../examples/megacpp/gb10_repro_bundle/run.sh)
- [loader.cpp](../examples/megacpp/gb10_repro_bundle/loader.cpp)
- [query_attrs.cpp](../examples/megacpp/gb10_repro_bundle/query_attrs.cpp)
- [kernel_baseline.cu](../examples/megacpp/gb10_repro_bundle/kernel_baseline.cu)
- [kernel_alloc_only.cu](../examples/megacpp/gb10_repro_bundle/kernel_alloc_only.cu)
- [kernel_sm100a.cu](../examples/megacpp/gb10_repro_bundle/kernel_sm100a.cu)
- [patch_elf.py](../examples/megacpp/gb10_repro_bundle/patch_elf.py)
- [patch_symbols.py](../examples/megacpp/gb10_repro_bundle/patch_symbols.py)
- [patch_nvinfo.py](../examples/megacpp/gb10_repro_bundle/patch_nvinfo.py)
- [README_gates.md](../examples/megacpp/gb10_repro_bundle/README_gates.md)
- [README_walkthrough.md](../examples/megacpp/gb10_repro_bundle/README_walkthrough.md)
- [public_claims.md](../examples/megacpp/gb10_repro_bundle/public_claims.md)

The receipts behind the article were collected on GB10 with CUDA 13.2 and driver 595.58.03. If your exact environment differs, treat the command flow as the stable part and the exact offsets or return paths as environment-specific.

## The baseline result in one screen

The narrow positive result is a trivial arithmetic kernel compiled for `sm_100a`, then patched so the ELF arch field says `sm_121a`.

The kernel is intentionally boring:

```cuda
extern "C" __global__ void k_baseline(int* out) {
    out[threadIdx.x] = threadIdx.x * 2 + 1;
}
```

That source is published in [kernel_baseline.cu](../examples/megacpp/gb10_repro_bundle/kernel_baseline.cu).

The public-safe receipt is the one that matters:

```text
# device: NVIDIA GB10  sm_121
[cuModuleLoadDataEx]                CUDA_SUCCESS
[cuLaunchKernel]                    CUDA_SUCCESS
[cuCtxSynchronize]                  CUDA_SUCCESS
# out[0..7]: 00000001 00000003 00000005 00000007 00000009 0000000b 0000000d 0000000f
```

That proves three specific things.

1. The user-space driver performs a software-visible architecture check at the cubin level.
2. GB10 will accept and execute at least some SASS originally emitted for the datacenter Blackwell line.
3. This is not the same thing as proving GB10 exposes working `tcgen05.mma`.

The third point is the one worth repeating, because it is where reverse-engineering writeups usually start to overclaim.

## Step 1: build the baseline cubin

From the bundle directory:

```bash
make clean
make build-baseline
```

Under the hood, the baseline build is equivalent to:

```bash
nvcc -arch=sm_100a --cubin -std=c++17 -lineinfo \
  -Xptxas -gno-tmem-access-check \
  kernel_baseline.cu -o kernel_baseline_100a.cubin
```

The `-gno-tmem-access-check` flag is harmless for the baseline kernel and keeps the command line aligned with the deeper probes that follow.

## Step 2: patch only the ELF arch field

The smallest patch in the whole workflow is [patch_elf.py](../examples/megacpp/gb10_repro_bundle/patch_elf.py). It rewrites only the low architecture bits in ELF `e_flags` and preserves the upper bits.

Run it directly:

```bash
./patch_elf.py kernel_baseline_100a.cubin kernel_baseline_patched.cubin sm_100a sm_121a
```

The source notes summarize the key field values like this:

- `sm_100a` -> low 16 bits `0x6402`
- `sm_121a` -> low 16 bits `0x7902`

So the patch is changing the architecture identity at the cubin metadata layer, not rewriting the kernel body.

If you want to inspect the result yourself:

```bash
readelf -h kernel_baseline_100a.cubin | grep -E "Flags|Machine"
readelf -h kernel_baseline_patched.cubin | grep -E "Flags|Machine"
```

## Step 3: load and run the patched cubin on GB10

The loader published in [loader.cpp](../examples/megacpp/gb10_repro_bundle/loader.cpp) uses the CUDA Driver API directly. It loads the cubin, resolves the kernel symbol, launches it, synchronizes, and prints the output buffer.

Build and run:

```bash
make run-baseline
```

Or run the loader explicitly:

```bash
g++ -O2 -std=c++17 -I/usr/local/cuda/include loader.cpp -o loader -L/usr/local/cuda/lib64 -lcuda
./loader kernel_baseline_patched.cubin k_baseline 32
```

This is the exact point where the public-safe positive claim ends. The patched baseline cubin executes. That is real. It is useful. It is still narrower than “GB10 has datacenter Blackwell tensor-path parity.”

## Step 4: move from baseline arithmetic to a tcgen05 probe

The next published probe is [kernel_alloc_only.cu](../examples/megacpp/gb10_repro_bundle/kernel_alloc_only.cu). It isolates `tcgen05.alloc` without mixing in the larger `mma` or TMA path.

The interesting part of that kernel is this block:

```cuda
if (threadIdx.x == 0) {
    uint32_t smem_ptr = __cvta_generic_to_shared(&tmem_addr);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
        :: "r"(smem_ptr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}
```

That is where the story stops being a simple arch-byte patch and becomes a gate walk.

Build the alloc-only cubin:

```bash
make build-alloc
```

Then try the first launch attempt:

```bash
./loader alloc_patched.cubin k_tcgen05_alloc 128
```

On the public-safe path, that does **not** complete as a working GB10 kernel.

## The four gates we found

The public repro bundle keeps the gate structure explicit because collapsing it into one sentence hides the engineering reality.

The four gates are:

1. ELF `e_flags` architecture validation
2. weak undefined reserved-SMEM symbols such as `.nv.reservedSmem.offset0` and `.nv.reservedSmem.cap`
3. `.nv.info.<kernel>` capability records
4. `.nv.capmerc.text.<kernel>` plus `.nv.merc.rela.*` signed capability metadata

You can also see this summarized in [README_gates.md](../examples/megacpp/gb10_repro_bundle/README_gates.md).

The important distinction is that the gates are not interchangeable.

- Gate 1 is simple identity metadata.
- Gate 2 is loader-side symbol plumbing.
- Gate 3 is mutable per-kernel metadata.
- Gate 4 is integrity-protected capability metadata.

That last transition is where naive cubin surgery stops being enough.

## Step 5: patch the reserved-SMEM symbols

The first `tcgen05.alloc` failure in the log moves from the arch gate to symbol resolution. The source notes report weak undefined symbols like these:

```text
.nv.reservedSmem.offset0
.nv.reservedSmem.cap
```

The published patcher for that stage is [patch_symbols.py](../examples/megacpp/gb10_repro_bundle/patch_symbols.py).

Run it in place on the alloc probe:

```bash
./patch_symbols.py alloc_patched.cubin alloc_patched.cubin \
  .nv.reservedSmem.offset0 .nv.reservedSmem.cap
```

Then retry the load:

```bash
./loader alloc_patched.cubin k_tcgen05_alloc 128
```

At this point the error changes, which tells you the earlier gate was real. It does **not** tell you that the final `tcgen05` path is now proven available.

## Step 6: strip selected `.nv.info` records

The next patcher in the bundle is [patch_nvinfo.py](../examples/megacpp/gb10_repro_bundle/patch_nvinfo.py). It removes selected `tcgen05`-specific records from `.nv.info.<kernel>`.

Run it like this:

```bash
./patch_nvinfo.py alloc_patched.cubin alloc_patched_info.cubin k_tcgen05_alloc
```

Then load the rewritten cubin:

```bash
./loader alloc_patched_info.cubin k_tcgen05_alloc 128
```

This is the exact step where the public-safe story still stops at `CUDA_ERROR_INVALID_IMAGE`. The deeper gate is still in the way.

## The exact point where the public-safe path stops

The clean public lane stops at gate 4: signed capability metadata in `.nv.capmerc.text.<kernel>` together with `.nv.merc.rela.*`.

That is why the conservative public wording is:

- the baseline arithmetic cubin executes after an arch-field patch;
- `tcgen05`-oriented probes hit additional driver-side gates;
- the clean public evidence does **not** prove working `tcgen05.mma` parity on GB10.

This is also why the bundle separates the parent repro lane from the deeper driver research lane. The public-safe lane is about what we can show cleanly with source files, patch scripts, and reproducible receipts. It is not a place to smuggle in a stronger silicon claim than the receipts support.

## One-command walkthrough

If you want the compact path instead of running each step manually, the bundle already includes [run.sh](../examples/megacpp/gb10_repro_bundle/run.sh):

```bash
./run.sh
```

That script runs:

1. `make all`
2. `./query_attrs`
3. `make run-baseline`
4. `make probe-alloc-gates`

The result is exactly the sequence this article describes: a working baseline, then a staged `tcgen05.alloc` gate walk that still stops before publication-grade proof.

## What works, and what does not

What works in the public-safe bundle:

- building an `sm_100a` baseline cubin;
- rewriting the ELF arch field to `sm_121a`;
- loading and executing the patched baseline on GB10;
- reproducing the staged failure movement for the alloc-only `tcgen05` probe;
- seeing that the driver path is layered rather than a single yes/no hardware switch.

What does **not** work in the public-safe bundle:

- a clean end-to-end `tcgen05.alloc` execute receipt on GB10;
- a clean end-to-end `tcgen05.mma` execute receipt on GB10;
- a public proof that GB10 physically exposes datacenter Blackwell `tcgen05` parity;
- a claim that deeper helper paths or routing knowledge inside `libcuda` are the same thing as runtime proof.

That distinction is the whole point of publishing the bundle this way.

## Why this article exists next to the other GB10 posts

The other two GB10 posts explain the meaning of the result and the difference between driver-visible hints and runtime proof:

- [What Our GB10 Experiments Actually Prove About Blackwell Consumer vs Datacenter Tensor Paths](./gb10-blackwell-tensor-paths-what-we-actually-proved.md)
- [Why Driver-Visible Paths Can Look Like Hardware Support on GB10, Even When Silicon Proof Is Missing](./gb10-driver-gates-and-false-capability-signals.md)

This article is the practical companion. It is here so another engineer can repeat the exact cubin patch, load path, and gate walk without guessing where the public evidence starts and where it stops.

## References

- [README.md](../examples/megacpp/gb10_repro_bundle/README.md)
- [README_walkthrough.md](../examples/megacpp/gb10_repro_bundle/README_walkthrough.md)
- [loader.cpp](../examples/megacpp/gb10_repro_bundle/loader.cpp)
- [kernel_baseline.cu](../examples/megacpp/gb10_repro_bundle/kernel_baseline.cu)
- [kernel_alloc_only.cu](../examples/megacpp/gb10_repro_bundle/kernel_alloc_only.cu)
- [patch_elf.py](../examples/megacpp/gb10_repro_bundle/patch_elf.py)
- [patch_symbols.py](../examples/megacpp/gb10_repro_bundle/patch_symbols.py)
- [patch_nvinfo.py](../examples/megacpp/gb10_repro_bundle/patch_nvinfo.py)
- [What Our GB10 Experiments Actually Prove About Blackwell Consumer vs Datacenter Tensor Paths](./gb10-blackwell-tensor-paths-what-we-actually-proved.md)
- [Why Driver-Visible Paths Can Look Like Hardware Support on GB10, Even When Silicon Proof Is Missing](./gb10-driver-gates-and-false-capability-signals.md)
