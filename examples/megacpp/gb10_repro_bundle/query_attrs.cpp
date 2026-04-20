// Query the CUDA device attributes most relevant to the GB10 receipts.
// This is useful both before and after any deeper driver patch experiment.

#include <cuda.h>
#include <cstdio>
#include <cstdint>

static const char* cu_err(CUresult r){const char*s=nullptr;cuGetErrorName(r,&s);return s?s:"?";}

struct attr { CUdevice_attribute id; const char* name; };
static attr ATTRS[] = {
    {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,                  "compute_cap_major"},
    {CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,                  "compute_cap_minor"},
    {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,               "max_smem_per_block"},
    {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,         "max_smem_per_block_optin"},
    {CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,      "max_smem_per_sm"},
    {CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,          "reserved_smem_per_block"},
    {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,                   "max_regs_per_block"},
    {CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,          "max_regs_per_sm"},
    {CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,                      "sm_count"},
    {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,                     "max_threads_per_block"},
    {CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,            "max_threads_per_sm"},
    {CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,             "max_blocks_per_sm"},
    {CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,                   "mem_bus_width"},
    {CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,                             "l2_cache_bytes"},
    {CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,                             "pci_device_id"},
    {CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,                            "cluster_launch_supported"},
    {CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED,     "deferred_mapping"},
};

int main(){
    cuInit(0);
    CUdevice d; cuDeviceGet(&d, 0);
    char name[128]={0}; cuDeviceGetName(name, sizeof(name), d);
    printf("# device: %s\n", name);
    for (size_t i = 0; i < sizeof(ATTRS)/sizeof(ATTRS[0]); ++i) {
        int v = -1;
        CUresult r = cuDeviceGetAttribute(&v, ATTRS[i].id, d);
        if (r != CUDA_SUCCESS) {
            printf("  %-32s  (err=%s)\n", ATTRS[i].name, cu_err(r));
        } else {
            printf("  %-32s  %d\n", ATTRS[i].name, v);
        }
    }
    return 0;
}
