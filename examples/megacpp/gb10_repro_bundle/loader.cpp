// CUDA Driver API loader for the GB10 reproduction bundle.
//
// Loads a patched .cubin (ELF e_flags rewritten from sm_100a to sm_121a),
// launches a selected kernel, synchronizes, and prints Driver API receipts.
//
// The narrow positive claim in this bundle is the baseline arithmetic kernel.
// Deeper tcgen05-oriented probes are exploratory and should be interpreted
// conservatively.

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static const char* cu_err(CUresult r){const char*s=nullptr;cuGetErrorName(r,&s);return s?s:"?";}
static const char* cu_msg(CUresult r){const char*s=nullptr;cuGetErrorString(r,&s);return s?s:"?";}
#define CHK(x) do { CUresult _r=(x); \
    fprintf(stderr,"[%-34s] %-30s  %s\n",#x,cu_err(_r),cu_msg(_r)); \
} while(0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <cubin> <kernel> [block=32] [grid=1] [clusterX=1] [smem_bytes=0]\n"
            "  ./loader kernel_baseline_patched.cubin k_baseline\n"
            "  ./loader alloc_patched.cubin k_tcgen05_alloc 128\n"
            "  ./loader kernel_sm100a_patched.cubin k_tcgen05_mma 128\n"
            "  ./loader kernel_sm100a_patched.cubin k_tma_multicast 128 1 2 16384\n",
            argv[0]);
        return 2;
    }
    const char* cubin = argv[1];
    const char* sym   = argv[2];
    int  block    = argc > 3 ? atoi(argv[3]) : 32;
    int  grid     = argc > 4 ? atoi(argv[4]) : 1;
    int  clusterX = argc > 5 ? atoi(argv[5]) : 1;
    unsigned smem = argc > 6 ? (unsigned)strtoul(argv[6], nullptr, 0) : 0;

    const unsigned SMEM_CAP = 99328;
    if (smem > SMEM_CAP) {
        fprintf(stderr,"# smem %u > GB10 cap %u — clamping\n", smem, SMEM_CAP);
        smem = SMEM_CAP;
    }

    CHK(cuInit(0));

    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    char name[128]={0}; cuDeviceGetName(name,sizeof(name),dev);
    int maj=0,min=0,smemOptin=0,smemStatic=0,smemPerBlock=0;
    cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&smemOptin,   CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    cuDeviceGetAttribute(&smemPerBlock,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,        dev);
    cuDeviceGetAttribute(&smemStatic,  CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,   dev);
    fprintf(stderr,"# device: %s  sm_%d%d\n", name, maj, min);
    fprintf(stderr,"# smem cap (default/optin/reserved): %d / %d / %d bytes\n",
            smemPerBlock, smemOptin, smemStatic);

    CUcontext ctx; CHK(cuCtxCreate(&ctx, nullptr, 0, dev));

    FILE* f = fopen(cubin, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", cubin); return 3; }
    fseek(f, 0, SEEK_END); long flen = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> blob(flen);
    if (fread(blob.data(), 1, flen, f) != (size_t)flen) { fclose(f); return 4; }
    fclose(f);

    char info_log[8192] = {0}, err_log[8192] = {0};
    CUjit_option opts[] = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE,
    };
    void* optv[] = {
        info_log, (void*)(uintptr_t)sizeof(info_log),
        err_log,  (void*)(uintptr_t)sizeof(err_log),
        (void*)(uintptr_t)1,
    };

    CUmodule mod;
    CUresult load_r = cuModuleLoadDataEx(&mod, blob.data(),
        sizeof(opts)/sizeof(opts[0]), opts, optv);
    fprintf(stderr,"[cuModuleLoadDataEx]                %-30s  %s\n",
            cu_err(load_r), cu_msg(load_r));
    if (info_log[0]) fprintf(stderr, "--- info log ---\n%s\n", info_log);
    if (err_log[0])  fprintf(stderr, "--- error log ---\n%s\n", err_log);
    if (load_r != CUDA_SUCCESS) { cuCtxDestroy(ctx); return 5; }

    CUfunction fn; CHK(cuModuleGetFunction(&fn, mod, sym));

    if (smem > 0) {
        CUresult r = cuFuncSetAttribute(fn,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem);
        fprintf(stderr,"[cuFuncSetAttribute(MAX_DYN_SMEM=%u)] %-12s  %s\n",
                smem, cu_err(r), cu_msg(r));
    }

    CUdeviceptr d_out; CHK(cuMemAlloc(&d_out, 4096));
    CHK(cuMemsetD8(d_out, 0, 4096));
    void* args[] = { &d_out };

    CUresult launch_r;
    if (clusterX > 1) {
        CUlaunchAttribute attrs[1] = {};
        attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attrs[0].value.clusterDim.x = clusterX;
        attrs[0].value.clusterDim.y = 1;
        attrs[0].value.clusterDim.z = 1;

        CUlaunchConfig cfg = {};
        cfg.gridDimX  = grid * clusterX;
        cfg.gridDimY  = 1;
        cfg.gridDimZ  = 1;
        cfg.blockDimX = block;
        cfg.blockDimY = 1;
        cfg.blockDimZ = 1;
        cfg.sharedMemBytes = smem;
        cfg.hStream   = 0;
        cfg.attrs     = attrs;
        cfg.numAttrs  = 1;
        launch_r = cuLaunchKernelEx(&cfg, fn, args, nullptr);
    } else {
        launch_r = cuLaunchKernel(fn,
            grid,1,1, block,1,1,
            smem, 0, args, nullptr);
    }
    fprintf(stderr,"[cuLaunchKernel]                    %-30s  %s\n",
            cu_err(launch_r), cu_msg(launch_r));

    CUresult sync_r = cuCtxSynchronize();
    fprintf(stderr,"[cuCtxSynchronize]                  %-30s  %s\n",
            cu_err(sync_r), cu_msg(sync_r));

    if (sync_r == CUDA_SUCCESS) {
        uint32_t buf[8] = {0};
        cuMemcpyDtoH(buf, d_out, sizeof(buf));
        fprintf(stderr,"# out[0..7]:");
        for (int i=0;i<8;i++) fprintf(stderr," %08x", buf[i]);
        fprintf(stderr,"\n");
    }

    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return sync_r == CUDA_SUCCESS ? 0 : 1;
}
