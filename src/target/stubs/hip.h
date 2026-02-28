/**
 * \file hip.h
 * \brief Stub library header for lazy loading ROCm/HIP libraries at runtime.
 *
 * This mirrors the existing CUDA stubs in src/target/stubs/:
 * - Instead of linking against libamdhip64.so at build time, TileLang can link
 *   against a small stub library (libhip_stub.so) that resolves HIP symbols via
 *   dlopen()/dlsym() on first use.
 *
 * This enables:
 * 1. Importing TileLang on CPU-only machines (no ROCm runtime installed).
 * 2. Avoiding version conflicts by using the ROCm runtime available at runtime.
 * 3. Building a single wheel that can run across environments.
 *
 * Usage:
 *   #include "target/stubs/hip.h"
 *   hipError_t e = hipSetDevice(0);
 */

#pragma once

// Define guard before including vendor/hip_runtime.h.
// This ensures vendor/hip_runtime.h can only be included through this stub
// header.
#define _TILELANG_HIP_STUB_INCLUDE_GUARD

// Prefer real ROCm headers when available for exact ABI matching. Fall back to
// a minimal vendored header to allow building CPU-only wheels.
#if __has_include(<hip/hip_runtime_api.h>)
#include <hip/hip_runtime_api.h>
#else
#include "vendor/hip_runtime.h"
#endif

#undef _TILELANG_HIP_STUB_INCLUDE_GUARD

// Symbol visibility macros for shared library export.
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef TILELANG_HIP_STUB_EXPORTS
#define TILELANG_HIP_STUB_API __declspec(dllexport)
#else
#define TILELANG_HIP_STUB_API __declspec(dllimport)
#endif
#else
#define TILELANG_HIP_STUB_API __attribute__((visibility("default")))
#endif

// X-macro for listing all required HIP API functions.
// Format: _(function_name)
//
// NOTE: libamdhip64.so contains both HIP "runtime" APIs and module/driver-style
// APIs (hipModuleLoadData, hipModuleLaunchKernel, ...). TVM's ROCm runtime uses
// both, so we stub them from the same shared library.
#define TILELANG_LIBHIP_API_REQUIRED(_)                                        \
  _(hipGetErrorName)                                                           \
  _(hipGetErrorString)                                                         \
  _(hipGetLastError)                                                           \
  _(hipSetDevice)                                                              \
  _(hipGetDevice)                                                              \
  _(hipGetDeviceCount)                                                         \
  _(hipDeviceGetAttribute)                                                     \
  _(hipDeviceGetName)                                                          \
  _(hipGetDeviceProperties)                                                    \
  _(hipMalloc)                                                                 \
  _(hipFree)                                                                   \
  _(hipHostMalloc)                                                             \
  _(hipHostFree)                                                               \
  _(hipMemcpy)                                                                 \
  _(hipMemcpyAsync)                                                            \
  _(hipMemcpyPeerAsync)                                                        \
  _(hipStreamCreate)                                                           \
  _(hipStreamDestroy)                                                          \
  _(hipStreamSynchronize)                                                      \
  _(hipEventCreate)                                                            \
  _(hipEventDestroy)                                                           \
  _(hipEventRecord)                                                            \
  _(hipEventSynchronize)                                                       \
  _(hipEventElapsedTime)                                                       \
  _(hipModuleLoadData)                                                         \
  _(hipModuleUnload)                                                           \
  _(hipModuleGetFunction)                                                      \
  _(hipModuleGetGlobal)                                                        \
  _(hipModuleLaunchKernel)

namespace tvm::tl::hip {

/**
 * \brief HIP API accessor struct with lazy loading support.
 *
 * Similar to tvm::tl::cuda::CUDADriverAPI, this struct resolves libamdhip64.so
 * symbols lazily on first use.
 *
 * Function pointer members have a trailing underscore to avoid collisions with
 * the exported global wrapper functions.
 */
struct TILELANG_HIP_STUB_API HIPDriverAPI {
  // Explicit function pointer types to avoid decltype issues with
  // C++ template overloads in HIP headers (e.g., hipMalloc).
  const char *(*hipGetErrorName_)(hipError_t);
  const char *(*hipGetErrorString_)(hipError_t);
  hipError_t (*hipGetLastError_)(void);
  hipError_t (*hipSetDevice_)(int);
  hipError_t (*hipGetDevice_)(int *);
  hipError_t (*hipGetDeviceCount_)(int *);
  hipError_t (*hipDeviceGetAttribute_)(int *, hipDeviceAttribute_t, int);
  hipError_t (*hipDeviceGetName_)(char *, int, int);
  hipError_t (*hipGetDeviceProperties_)(hipDeviceProp_t *, int);
  hipError_t (*hipMalloc_)(void **, size_t);
  hipError_t (*hipFree_)(void *);
  hipError_t (*hipHostMalloc_)(void **, size_t, unsigned int);
  hipError_t (*hipHostFree_)(void *);
  hipError_t (*hipMemcpy_)(void *, const void *, size_t, hipMemcpyKind);
  hipError_t (*hipMemcpyAsync_)(void *, const void *, size_t, hipMemcpyKind,
                                hipStream_t);
  hipError_t (*hipMemcpyPeerAsync_)(void *, int, const void *, int, size_t,
                                    hipStream_t);
  hipError_t (*hipStreamCreate_)(hipStream_t *);
  hipError_t (*hipStreamDestroy_)(hipStream_t);
  hipError_t (*hipStreamSynchronize_)(hipStream_t);
  hipError_t (*hipEventCreate_)(hipEvent_t *);
  hipError_t (*hipEventDestroy_)(hipEvent_t);
  hipError_t (*hipEventRecord_)(hipEvent_t, hipStream_t);
  hipError_t (*hipEventSynchronize_)(hipEvent_t);
  hipError_t (*hipEventElapsedTime_)(float *, hipEvent_t, hipEvent_t);
  hipError_t (*hipModuleLoadData_)(hipModule_t *, const void *);
  hipError_t (*hipModuleUnload_)(hipModule_t);
  hipError_t (*hipModuleGetFunction_)(hipFunction_t *, hipModule_t,
                                      const char *);
  hipError_t (*hipModuleGetGlobal_)(hipDeviceptr_t *, size_t *, hipModule_t,
                                    const char *);
  hipError_t (*hipModuleLaunchKernel_)(hipFunction_t, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       hipStream_t, void **, void **);

  static HIPDriverAPI *get();
  static bool is_available();
  static void *get_handle();
};

} // namespace tvm::tl::hip

// ============================================================================
// Global wrapper functions for lazy-loaded HIP API
// ============================================================================
// These functions provide drop-in replacements for HIP runtime/module API
// calls. The implementations are in hip.cc.

extern "C" {

TILELANG_HIP_STUB_API const char *hipGetErrorName(hipError_t error);
TILELANG_HIP_STUB_API const char *hipGetErrorString(hipError_t error);
TILELANG_HIP_STUB_API hipError_t hipGetLastError(void);

TILELANG_HIP_STUB_API hipError_t hipSetDevice(int deviceId);
TILELANG_HIP_STUB_API hipError_t hipGetDevice(int *deviceId);
TILELANG_HIP_STUB_API hipError_t hipGetDeviceCount(int *count);

TILELANG_HIP_STUB_API hipError_t
hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId);
TILELANG_HIP_STUB_API hipError_t hipDeviceGetName(char *name, int len,
                                                  int deviceId);
TILELANG_HIP_STUB_API hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop,
                                                        int deviceId);

TILELANG_HIP_STUB_API hipError_t hipMalloc(void **ptr, size_t size);
TILELANG_HIP_STUB_API hipError_t hipFree(void *ptr);
TILELANG_HIP_STUB_API hipError_t hipHostMalloc(void **ptr, size_t size,
                                               unsigned int flags);
TILELANG_HIP_STUB_API hipError_t hipHostFree(void *ptr);

TILELANG_HIP_STUB_API hipError_t hipMemcpy(void *dst, const void *src,
                                           size_t sizeBytes,
                                           hipMemcpyKind kind);
TILELANG_HIP_STUB_API hipError_t hipMemcpyAsync(void *dst, const void *src,
                                                size_t sizeBytes,
                                                hipMemcpyKind kind,
                                                hipStream_t stream);
TILELANG_HIP_STUB_API hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId,
                                                    const void *src,
                                                    int srcDeviceId,
                                                    size_t sizeBytes,
                                                    hipStream_t stream);

TILELANG_HIP_STUB_API hipError_t hipStreamCreate(hipStream_t *stream);
TILELANG_HIP_STUB_API hipError_t hipStreamDestroy(hipStream_t stream);
TILELANG_HIP_STUB_API hipError_t hipStreamSynchronize(hipStream_t stream);

TILELANG_HIP_STUB_API hipError_t hipEventCreate(hipEvent_t *event);
TILELANG_HIP_STUB_API hipError_t hipEventDestroy(hipEvent_t event);
TILELANG_HIP_STUB_API hipError_t hipEventRecord(hipEvent_t event,
                                                hipStream_t stream);
TILELANG_HIP_STUB_API hipError_t hipEventSynchronize(hipEvent_t event);
TILELANG_HIP_STUB_API hipError_t hipEventElapsedTime(float *ms,
                                                     hipEvent_t start,
                                                     hipEvent_t stop);

TILELANG_HIP_STUB_API hipError_t hipModuleLoadData(hipModule_t *module,
                                                   const void *image);
TILELANG_HIP_STUB_API hipError_t hipModuleUnload(hipModule_t module);
TILELANG_HIP_STUB_API hipError_t hipModuleGetFunction(hipFunction_t *function,
                                                      hipModule_t module,
                                                      const char *name);
TILELANG_HIP_STUB_API hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr,
                                                    size_t *bytes,
                                                    hipModule_t module,
                                                    const char *name);
TILELANG_HIP_STUB_API hipError_t hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void **kernelParams, void **extra);

} // extern "C"
