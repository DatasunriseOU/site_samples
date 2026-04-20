# GB10 Gate Matrix

The public-safe GB10 result is not one claim. It is a sequence of gates.

| Gate | Surface | Error before bypass | Bypass in this bundle | What it proves |
| --- | --- | --- | --- | --- |
| 1 | ELF `e_flags` arch check | `CUDA_ERROR_NO_BINARY_FOR_GPU` | `patch_elf.py` | GB10 accepts at least some `sm_100a` SASS after the architecture field is rewritten |
| 2 | weak UND reserved-SMEM symbols | `CUDA_ERROR_NOT_FOUND` | `patch_symbols.py` | the driver exposes another software gate before the kernel reaches normal execution |
| 3 | `.nv.info.<kernel>` capability records | `CUDA_ERROR_INVALID_IMAGE` | `patch_nvinfo.py` | more metadata gating exists beyond ELF and symbol resolution |
| 4 | `.nv.capmerc.*` and `.nv.merc.rela.*` | `CUDA_ERROR_INVALID_IMAGE` | no public-safe bypass here | the public evidence set stops at integrity-protected capability metadata |

Scripts involved in the public-safe lane:

- [`patch_elf.py`](./patch_elf.py)
- [`patch_symbols.py`](./patch_symbols.py)
- [`patch_nvinfo.py`](./patch_nvinfo.py)
- [`loader.cpp`](./loader.cpp)
- [`query_attrs.cpp`](./query_attrs.cpp)

## The narrow positive result

This is the cleanest receipt in the bundle:

```text
[cuModuleLoadDataEx]                CUDA_SUCCESS
[cuLaunchKernel]                    CUDA_SUCCESS
[cuCtxSynchronize]                  CUDA_SUCCESS
# out[0..7]: 00000001 00000003 00000005 00000007 00000009 0000000b 0000000d 0000000f
```

That result came from:

1. compiling `kernel_baseline.cu` for `sm_100a`;
2. rewriting only the ELF `e_flags` low 16 bits to `sm_121a`;
3. loading and launching the patched cubin through `loader.cpp`.

The exact baseline sources for that proof are:

- [`kernel_baseline.cu`](./kernel_baseline.cu)
- [`patch_elf.py`](./patch_elf.py)
- [`loader.cpp`](./loader.cpp)

## What works

- baseline arithmetic SASS after the ELF arch rewrite;
- the staged gate walk itself, which shows the failures move as each software
  barrier is patched.

In practice, the easiest way to repeat that is:

```bash
make run-baseline
```

## What still does not have a clean public execute receipt

- `tcgen05.alloc`
- `tcgen05.ld`
- `tcgen05.mma`
- TMA multicast cluster probes

Those probe sources are still included for reproducibility:

- [`kernel_alloc_only.cu`](./kernel_alloc_only.cu)
- [`kernel_sm100a.cu`](./kernel_sm100a.cu)

That is why the public wording stays narrow: the evidence shows layered
software gating and a real baseline execution result, but not a clean
publication-grade `tcgen05` completion receipt.

## What belongs in the deeper lane instead

Anything that depends on copied-`libcuda` patching, helper-cubin redirection,
or changing how the driver selects its internal capability path belongs under
[`driver_patch_lane/`](./driver_patch_lane/), not in the public-safe proof.
