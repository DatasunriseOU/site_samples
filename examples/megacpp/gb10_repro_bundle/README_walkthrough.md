# GB10 Walkthrough

This walkthrough mirrors the internal `smtest` flow, but keeps the public
interpretation conservative.

If you only want the shortest path, run [`run.sh`](./run.sh). If you want the
actual source and scripts, keep these files open while you work:

- [`loader.cpp`](./loader.cpp)
- [`query_attrs.cpp`](./query_attrs.cpp)
- [`kernel_baseline.cu`](./kernel_baseline.cu)
- [`kernel_alloc_only.cu`](./kernel_alloc_only.cu)
- [`kernel_sm100a.cu`](./kernel_sm100a.cu)
- [`patch_elf.py`](./patch_elf.py)
- [`patch_symbols.py`](./patch_symbols.py)
- [`patch_nvinfo.py`](./patch_nvinfo.py)

## 1. Build the source artifacts

```bash
make all
```

That produces:

- `kernel_baseline_100a.cubin`
- `kernel_baseline_patched.cubin`
- `alloc_100a.cubin`
- `alloc_patched.cubin`
- `kernel_sm100a.cubin`
- `kernel_sm100a_patched.cubin`
- `loader`
- `query_attrs`

## 2. Inspect what the driver reports

```bash
./query_attrs
```

This is the clean baseline for the host before any deeper patch story.

The readback source is [`query_attrs.cpp`](./query_attrs.cpp).

## 3. Run the narrow positive proof

```bash
make run-baseline
```

This should show:

- `cuModuleLoadDataEx == CUDA_SUCCESS`
- `cuLaunchKernel == CUDA_SUCCESS`
- `cuCtxSynchronize == CUDA_SUCCESS`
- output values `1, 3, 5, ... 15`

Interpretation:

- GB10 accepted and executed a baseline cubin that was originally compiled for
  `sm_100a`;
- this proves a loader/runtime fact, not full datacenter-path parity.

The three files behind this step are:

- [`kernel_baseline.cu`](./kernel_baseline.cu)
- [`patch_elf.py`](./patch_elf.py)
- [`loader.cpp`](./loader.cpp)

## 4. Walk the minimal `tcgen05.alloc` lane

```bash
make build-alloc
./loader alloc_patched.cubin k_tcgen05_alloc 128 || true
./patch_symbols.py alloc_patched.cubin alloc_patched.cubin \
  .nv.reservedSmem.offset0 .nv.reservedSmem.cap
./loader alloc_patched.cubin k_tcgen05_alloc 128 || true
./patch_nvinfo.py alloc_patched.cubin alloc_patched_info.cubin k_tcgen05_alloc
./loader alloc_patched_info.cubin k_tcgen05_alloc 128 || true
```

Expected shape of the walk:

1. before symbol patching: `CUDA_ERROR_NOT_FOUND`
2. after symbol patching: `CUDA_ERROR_INVALID_IMAGE`
3. after `.nv.info` stripping: still `CUDA_ERROR_INVALID_IMAGE`

Interpretation:

- the early gates are software and metadata driven;
- the public lane still stops before a clean `tcgen05` completion receipt.

The exact artifacts for this step are:

- [`kernel_alloc_only.cu`](./kernel_alloc_only.cu)
- [`patch_symbols.py`](./patch_symbols.py)
- [`patch_nvinfo.py`](./patch_nvinfo.py)
- [`loader.cpp`](./loader.cpp)

## 5. Optional fuller probes

```bash
make run-full-alloc || true
make run-full-mma || true
make run-full-tma || true
```

These targets are included because the articles discuss them, not because they
already establish public proof.

The fuller exploratory source is [`kernel_sm100a.cu`](./kernel_sm100a.cu).

## 6. If you want the deeper driver patch lane

Read [`driver_patch_lane/README.md`](./driver_patch_lane/README.md) and then
[`driver_patch_lane/patch_libcuda.py`](./driver_patch_lane/patch_libcuda.py).

That lane is user-space `libcuda` research and should be treated separately
from the public-safe baseline and gate-walk claims.

## 7. Recommended writeup discipline

Before publishing or summarizing the results, re-read
[`public_claims.md`](./public_claims.md). The point of this bundle is not only
to make the steps repeatable, but also to keep the wording tied to the exact
stage of proof you actually reached.
