// Isolate a single tcgen05 probe. No TMA, no cluster, no MMA.
// This keeps the public gate walk as small as possible.

#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void k_tcgen05_alloc(uint32_t* out) {
    __shared__ uint32_t tmem_addr;

    if (threadIdx.x == 0) {
        uint32_t smem_ptr = __cvta_generic_to_shared(&tmem_addr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
            :: "r"(smem_ptr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    __syncthreads();
    if (threadIdx.x == 0) out[0] = tmem_addr;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
            :: "r"(tmem_addr));
    }
}
