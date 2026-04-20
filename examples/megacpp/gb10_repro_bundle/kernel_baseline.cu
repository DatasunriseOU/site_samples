// Minimal sm_100a cubin: only baseline arithmetic, no tcgen05.
// This is the narrow positive result in the GB10 public evidence set.

#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void k_baseline(int* out) {
    out[threadIdx.x] = threadIdx.x * 2 + 1;
}
