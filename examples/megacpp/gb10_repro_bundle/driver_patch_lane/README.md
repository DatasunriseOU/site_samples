# GB10 Driver Patch Lane

This directory carries the deeper user-space `libcuda` research artifact for
the GB10 work.

What it is:

- a research-only path that patches a copy of `libcuda.so`;
- useful for reproducing the deeper helper-cubin and compute-capability-table
  routing experiments;
- intentionally separated from the public-safe baseline and gate-walk bundle.

What it is **not**:

- a proof that GB10 has shipping `tcgen05` parity with B200 or GB100;
- a statement that a deeper helper path is the same thing as an end-to-end
  execute receipt;
- a recommendation to replace the system driver in normal environments.

## Included file

- `patch_libcuda.py`: copy-patch helper for experimenting on a copied
  `libcuda.so` and loading it via `LD_LIBRARY_PATH`

## Safe way to use it

1. work on a copy of the driver;
2. prefer `--dry-run` first;
3. load the patched copy only through `LD_LIBRARY_PATH`;
4. compare the result against the baseline and gate-walk lane in the parent
   directory;
5. keep the public conclusion narrow unless you have an exact runtime receipt.
