# Public Claims Guardrail

Use this file as the last checkpoint before publishing issue comments, README
summaries, or blog copy derived from this bundle.

Use these phrases:

- `baseline sm_100a SASS executed on GB10 after an ELF arch-field rewrite`
- `multiple software gates exist before tcgen05-oriented probes can run`
- `driver-visible helpers and capability metadata are not runtime proof`
- `the current public evidence set leaves tcgen05 and TMEM on GB10 unproven`

Do not use these phrases:

- `6-byte patch unlocks tcgen05 on GB10`
- `GB10 has proven full B200 or GB100 tcgen05 parity`
- `the current public evidence conclusively proves physical presence or absence`
- `the driver patch lane is equivalent to a clean shipping support receipt`

## Internal discipline

Keep these statements separate:

1. the driver accepted the image;
2. helper routing changed;
3. a deeper user-space path was reached;
4. the exact instruction family completed and produced the expected output.

Only statement 4 is an end-to-end runtime proof.

## File-to-claim mapping

- [`kernel_baseline.cu`](./kernel_baseline.cu) + [`patch_elf.py`](./patch_elf.py)
  + [`loader.cpp`](./loader.cpp) support the narrow baseline execution claim.
- [`kernel_alloc_only.cu`](./kernel_alloc_only.cu) + [`patch_symbols.py`](./patch_symbols.py)
  + [`patch_nvinfo.py`](./patch_nvinfo.py) support the staged gate-walk claim.
- [`kernel_sm100a.cu`](./kernel_sm100a.cu) supports exploratory discussion, not
  a stronger public proof.
- [`driver_patch_lane/patch_libcuda.py`](./driver_patch_lane/patch_libcuda.py)
  belongs to the deeper research lane and should not be collapsed into the
  public-safe result.

## Short version

If you want one sentence that stays safe, use this shape:

`We can reproduce a baseline sm_100a -> sm_121a cubin patch that executes on GB10, and we can reproduce the later tcgen05 gate walk, but the public evidence still stops short of a clean tcgen05 runtime proof.`

If you want the deeper lane, say so explicitly and link the separate directory.
Do not merge the two stories.
