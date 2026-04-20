// Exploratory sm_100a kernel battery for the GB10 research lane.
// These kernels are useful for reproducing the deeper probe surface, but they
// are not publication-grade proof of tcgen05 support by themselves.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

__global__ void k_baseline(int* out) {
    out[threadIdx.x] = threadIdx.x * 2 + 1;
}

__global__ void k_tcgen05_alloc(uint32_t* out) {
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

__global__ void k_tcgen05_ld(uint32_t* out) {
    __shared__ uint32_t tmem_addr;

    if (threadIdx.x == 0) {
        uint32_t smem_ptr = __cvta_generic_to_shared(&tmem_addr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
            :: "r"(smem_ptr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    __syncthreads();

    uint32_t v0=0, v1=0, v2=0, v3=0;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
        : "r"(tmem_addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");

    out[threadIdx.x * 4 + 0] = v0;
    out[threadIdx.x * 4 + 1] = v1;
    out[threadIdx.x * 4 + 2] = v2;
    out[threadIdx.x * 4 + 3] = v3;

    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
            :: "r"(tmem_addr));
    }
}

__global__ void k_tcgen05_mma(uint32_t* out) {
    __shared__ __align__(128) uint8_t sA[64*16*2];
    __shared__ __align__(128) uint8_t sB[16*64*2];
    __shared__ uint32_t tmem_d;

    if (threadIdx.x == 0) {
        uint32_t smem_ptr = __cvta_generic_to_shared(&tmem_d);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n"
            :: "r"(smem_ptr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    __syncthreads();

    uint32_t sA_addr = __cvta_generic_to_shared(sA);
    uint32_t sB_addr = __cvta_generic_to_shared(sB);
    uint64_t descA = ((uint64_t)(sA_addr >> 4) & 0x3FFFF);
    uint64_t descB = ((uint64_t)(sB_addr >> 4) & 0x3FFFF);
    uint32_t idesc = 0;

    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
            :: "r"(tmem_d), "l"(descA), "l"(descB), "r"(idesc));
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                     :: "r"((uint32_t)0));
    }
    __syncthreads();

    if (threadIdx.x == 0) out[0] = tmem_d;

    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n"
            :: "r"(tmem_d));
    }
}

__global__ void __cluster_dims__(2, 1, 1) k_tma_multicast(
    const __grid_constant__ CUtensorMap tmap, int* out)
{
    extern __shared__ __align__(128) int smem[];
    __shared__ __align__(8) uint64_t bar;

    uint64_t tmap_ptr;
    asm volatile("mov.u64 %0, %1;" : "=l"(tmap_ptr) : "l"(&tmap));

    if (threadIdx.x == 0) {
        uint32_t bar_sptr = __cvta_generic_to_shared(&bar);
        uint32_t smem_sptr = __cvta_generic_to_shared(smem);

        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(bar_sptr));
        asm volatile("fence.proxy.async.shared::cta;\n");

        uint16_t mask = 0x3;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes.multicast::cluster "
            "[%0], [%1, {%2, %3}], [%4], %5;\n"
            :: "r"(smem_sptr), "l"(tmap_ptr), "r"(0), "r"(0),
               "r"(bar_sptr), "h"(mask));

        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 256;\n"
            :: "r"(bar_sptr));
    }
    __syncthreads();

    if (threadIdx.x < 32) out[threadIdx.x] = smem[threadIdx.x];
}

} // extern "C"
