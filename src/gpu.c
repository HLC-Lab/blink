/* Vendor wrapper: the only file in Blink that includes CUDA / HIP headers.
 * BLINK_GPU_CUDA or BLINK_GPU_HIP is defined by CMake for GPU builds; with
 * neither defined every call except gpu_available/gpu_backend_name/
 * gpu_device_count is fatal ("built without GPU support").                   */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "gpu.h"

#if defined(BLINK_GPU_CUDA)
#  include <cuda_runtime_api.h>
#  define GPU_BACKEND_NAME        "CUDA"
#  define gpuError_t              cudaError_t
#  define gpuSuccess              cudaSuccess
#  define gpuGetErrorString       cudaGetErrorString
#  define gpuGetDeviceCount       cudaGetDeviceCount
#  define gpuSetDevice            cudaSetDevice
#  define gpuMalloc               cudaMalloc
#  define gpuFree                 cudaFree
#  define gpuMemset               cudaMemset
#  define gpuMemcpy               cudaMemcpy
#  define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#  define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#  define gpuDeviceSynchronize    cudaDeviceSynchronize
#elif defined(BLINK_GPU_HIP)
#  include <hip/hip_runtime_api.h>
#  define GPU_BACKEND_NAME        "HIP"
#  define gpuError_t              hipError_t
#  define gpuSuccess              hipSuccess
#  define gpuGetErrorString       hipGetErrorString
#  define gpuGetDeviceCount       hipGetDeviceCount
#  define gpuSetDevice            hipSetDevice
#  define gpuMalloc               hipMalloc
#  define gpuFree                 hipFree
#  define gpuMemset               hipMemset
#  define gpuMemcpy               hipMemcpy
#  define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#  define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#  define gpuDeviceSynchronize    hipDeviceSynchronize
#endif

static void gpu_fail(const char *what, const char *why)
{
    fprintf(stderr, "GPU error in %s: %s\n", what, why);
    int init = 0;
    MPI_Initialized(&init);
    if (init) MPI_Abort(MPI_COMM_WORLD, -1);
    exit(1);
}

#ifdef GPU_BACKEND_NAME

#define GPU_CHECK(call) do {                                            \
        gpuError_t e_ = (call);                                         \
        if (e_ != gpuSuccess) gpu_fail(#call, gpuGetErrorString(e_));   \
    } while (0)

int         gpu_available(void)    { return 1; }
const char *gpu_backend_name(void) { return GPU_BACKEND_NAME; }

int gpu_device_count(void)
{
    int n = 0;
    if (gpuGetDeviceCount(&n) != gpuSuccess) return 0;
    return n;
}

void gpu_set_device(int dev) { GPU_CHECK(gpuSetDevice(dev)); }

void *gpu_malloc(size_t n)
{
    void *p = NULL;
    GPU_CHECK(gpuMalloc(&p, n ? n : 1));
    return p;
}

void gpu_free(void *p)                                { GPU_CHECK(gpuFree(p)); }
void gpu_memset(void *dst, int value, size_t n)       { GPU_CHECK(gpuMemset(dst, value, n)); GPU_CHECK(gpuDeviceSynchronize()); }
/* sync after h2d: a pageable-memory cudaMemcpy may return before the DMA finished, and a peer
 * (CUDA IPC / UCX copy stream) could then read stale bytes.  All h2d copies are outside timed regions. */
void gpu_memcpy_h2d(void *dst, const void *src, size_t n) { GPU_CHECK(gpuMemcpy(dst, src, n, gpuMemcpyHostToDevice)); GPU_CHECK(gpuDeviceSynchronize()); }
void gpu_memcpy_d2h(void *dst, const void *src, size_t n) { GPU_CHECK(gpuMemcpy(dst, src, n, gpuMemcpyDeviceToHost)); }
void gpu_memcpy_d2d(void *dst, const void *src, size_t n) { GPU_CHECK(gpuMemcpy(dst, src, n, gpuMemcpyDeviceToDevice)); GPU_CHECK(gpuDeviceSynchronize()); }
void gpu_sync(void)                                   { GPU_CHECK(gpuDeviceSynchronize()); }

#else  /* no backend compiled in */

#define NO_GPU() gpu_fail(__func__, "built without GPU support; reconfigure with -DBLINK_GPU=CUDA or -DBLINK_GPU=HIP")

int         gpu_available(void)    { return 0; }
const char *gpu_backend_name(void) { return "none"; }
int         gpu_device_count(void) { return 0; }
void        gpu_set_device(int dev) { (void)dev; NO_GPU(); }
void       *gpu_malloc(size_t n)    { (void)n; NO_GPU(); return NULL; }
void        gpu_free(void *p)      { (void)p; NO_GPU(); }
void        gpu_memset(void *d, int v, size_t n)               { (void)d; (void)v; (void)n; NO_GPU(); }
void        gpu_memcpy_h2d(void *d, const void *s, size_t n)   { (void)d; (void)s; (void)n; NO_GPU(); }
void        gpu_memcpy_d2h(void *d, const void *s, size_t n)   { (void)d; (void)s; (void)n; NO_GPU(); }
void        gpu_memcpy_d2d(void *d, const void *s, size_t n)   { (void)d; (void)s; (void)n; NO_GPU(); }
void        gpu_sync(void)         { NO_GPU(); }

#endif
