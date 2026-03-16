// Minimal HIP runtime/driver API declarations for TileLang's HIP stub library.
//
// This file exists to allow building the stub (src/target/stubs/hip.cc) without
// requiring a full ROCm SDK at build time. When ROCm headers are available,
// target/stubs/hip.h prefers including <hip/hip_runtime_api.h>.
//
// IMPORTANT:
// - This header is NOT a complete HIP API.
// - Types that are passed by pointer are kept opaque/incomplete on purpose.
// - Do not include this file directly; include target/stubs/hip.h instead.

// Guard to ensure this header is only included by the stub wrapper header.
#ifndef _TILELANG_HIP_STUB_INCLUDE_GUARD
#error "vendor/hip_runtime.h should only be included by target/stubs/hip.h. "     \
    "Do not include this file directly; include target/stubs/hip.h instead."
#endif

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Opaque handle types (match HIP's public API shape).
// -----------------------------------------------------------------------------
typedef struct ihipStream_t *hipStream_t;
typedef struct ihipEvent_t *hipEvent_t;
typedef struct ihipModule_t *hipModule_t;
typedef struct ihipFunction_t *hipFunction_t;

// In HIP, hipDeviceptr_t is an opaque device pointer type.  We only ever pass
// it around, never dereference it from host code.
typedef void *hipDeviceptr_t;

// In the full ROCm SDK this is a large struct.  The stub never needs its
// contents, only the pointer type.
typedef struct hipDeviceProp_t hipDeviceProp_t;

// -----------------------------------------------------------------------------
// Scalar enums / values.
// -----------------------------------------------------------------------------
// Keep these as int-compatible so calling conventions match even when using the
// vendor fallback header.
typedef enum hipError_t {
  hipSuccess = 0,
  hipErrorDeinitialized = 3,
  hipErrorUnknown = 999
} hipError_t;

// hipMemcpyKind is passed by value.  Values follow the CUDA convention.
typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;

// hipDeviceAttribute_t is passed by value.  We do not rely on specific numeric
// values in the stub.  Use the real ROCm headers when available.
typedef int hipDeviceAttribute_t;

// -----------------------------------------------------------------------------
// Minimal subset of HIP runtime/driver entrypoints used by TileLang/TVM.
// -----------------------------------------------------------------------------
const char *hipGetErrorName(hipError_t error);
const char *hipGetErrorString(hipError_t error);
hipError_t hipGetLastError(void);

hipError_t hipSetDevice(int deviceId);
hipError_t hipGetDevice(int *deviceId);
hipError_t hipGetDeviceCount(int *count);

hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId);
hipError_t hipDeviceGetName(char *name, int len, int deviceId);
hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId);

hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipFree(void *ptr);
hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void *ptr);

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src,
                              int srcDeviceId, size_t sizeBytes,
                              hipStream_t stream);

hipError_t hipStreamCreate(hipStream_t *stream);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);

hipError_t hipEventCreate(hipEvent_t *event);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);

hipError_t hipModuleLoadData(hipModule_t *module, const void *image);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module,
                                const char *name);
hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t module, const char *name);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams,
                                 void **extra);

#ifdef __cplusplus
} // extern "C"
#endif
