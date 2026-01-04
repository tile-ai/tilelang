/**
 * \file cuda.h
 * \brief Stub library for lazy loading libcuda.so at runtime.
 *
 * This library provides drop-in replacements for CUDA driver API functions.
 * It allows tilelang to be imported on CPU-only machines without CUDA
 * installed. The actual libcuda.so is loaded lazily on first API call.
 *
 * Usage:
 *
 * 1. Link against libcuda_stub.so instead of libcuda.so
 *
 * 2. Call CUDA driver API functions normally - they are provided as
 *    exported global functions with C linkage:
 *
 *    ```cpp
 *    #include "target/stubs/cuda.h"
 *    CUresult result = cuModuleLoadData(&mod, image);
 *    ```
 *
 * 3. For advanced use, access the singleton directly:
 *
 *    ```cpp
 *    auto* api = tvm::tl::cuda::CUDADriverAPI::get();
 *    bool available = tvm::tl::cuda::CUDADriverAPI::is_available();
 *    ```
 */

#pragma once

// Define guard before including vendor/cuda.h
// This ensures vendor/cuda.h can only be included through this stub header.
#define _TILELANG_CUDA_STUB_INCLUDE_GUARD

#include "vendor/cuda.h" // include the full CUDA driver API types

#undef _TILELANG_CUDA_STUB_INCLUDE_GUARD

// Symbol visibility macros for shared library export
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef TILELANG_CUDA_STUB_EXPORTS
#define TILELANG_CUDA_STUB_API __declspec(dllexport)
#else
#define TILELANG_CUDA_STUB_API __declspec(dllimport)
#endif
#else
#define TILELANG_CUDA_STUB_API __attribute__((visibility("default")))
#endif

// X-macro for listing all required CUDA driver API functions.
// Format: _(function_name)
// These are the core functions used by TVM/tilelang CUDA runtime.
#define TILELANG_LIBCUDA_API_REQUIRED(_)                                       \
  _(cuGetErrorName)                                                            \
  _(cuGetErrorString)                                                          \
  _(cuCtxGetDevice)                                                            \
  _(cuCtxGetLimit)                                                             \
  _(cuCtxSetLimit)                                                             \
  _(cuCtxResetPersistingL2Cache)                                               \
  _(cuDeviceGetName)                                                           \
  _(cuDeviceGetAttribute)                                                      \
  _(cuModuleLoadData)                                                          \
  _(cuModuleLoadDataEx)                                                        \
  _(cuModuleUnload)                                                            \
  _(cuModuleGetFunction)                                                       \
  _(cuModuleGetGlobal)                                                         \
  _(cuFuncSetAttribute)                                                        \
  _(cuLaunchKernel)                                                            \
  _(cuLaunchKernelEx)                                                          \
  _(cuLaunchCooperativeKernel)                                                 \
  _(cuMemsetD32)                                                               \
  _(cuStreamSetAttribute)

// Optional APIs (may not exist in older drivers or specific configurations)
// These are loaded but may be nullptr if not available
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
#define TILELANG_LIBCUDA_API_OPTIONAL(_)                                       \
  _(cuTensorMapEncodeTiled)                                                    \
  _(cuTensorMapEncodeIm2col)
#else
#define TILELANG_LIBCUDA_API_OPTIONAL(_)
#endif

namespace tvm::tl::cuda {

/**
 * \brief CUDA Driver API accessor struct with lazy loading support.
 *
 * This struct provides lazy loading of libcuda.so symbols at first use,
 * allowing tilelang to be imported on machines without CUDA installed.
 * The library handle and function pointers are stored as static members
 * to ensure one-time initialization.
 *
 * Usage:
 *   CUresult result = CUDADriverAPI::get()->cuModuleLoadData_(&module, image);
 *
 * Note: Function pointers have a trailing underscore to differentiate from
 * the macro-redefined names in cuda.h (e.g., cuModuleGetGlobal ->
 * cuModuleGetGlobal_v2)
 */
struct TILELANG_CUDA_STUB_API CUDADriverAPI {
// Create function pointer members for each API function
// The trailing underscore avoids conflict with cuda.h macros
#define CREATE_MEMBER(name) decltype(&name) name##_;
  TILELANG_LIBCUDA_API_REQUIRED(CREATE_MEMBER)
  TILELANG_LIBCUDA_API_OPTIONAL(CREATE_MEMBER)
#undef CREATE_MEMBER

  /**
   * \brief Get the singleton instance of CUDADriverAPI.
   *
   * On first call, this loads libcuda.so and resolves all symbols.
   * Subsequent calls return the cached instance.
   *
   * \return Pointer to the singleton CUDADriverAPI instance.
   * \throws std::runtime_error if libcuda.so cannot be loaded or
   *         required symbols are missing.
   */
  static CUDADriverAPI *get();

  /**
   * \brief Check if CUDA driver is available without throwing.
   *
   * \return true if libcuda.so can be loaded, false otherwise.
   */
  static bool is_available();

  /**
   * \brief Get the raw library handle for libcuda.so.
   *
   * \return The dlopen handle, or nullptr if not loaded.
   */
  static void *get_handle();
};

} // namespace tvm::tl::cuda

// ============================================================================
// Global wrapper functions for lazy-loaded CUDA driver API
// ============================================================================
// These functions provide drop-in replacements for CUDA driver API calls.
// They are exported from the stub library and can be linked against directly.
// The implementations are in cuda.cc.

extern "C" {

TILELANG_CUDA_STUB_API CUresult cuGetErrorName(CUresult error,
                                               const char **pStr);
TILELANG_CUDA_STUB_API CUresult cuGetErrorString(CUresult error,
                                                 const char **pStr);
TILELANG_CUDA_STUB_API CUresult cuCtxGetDevice(CUdevice *device);
TILELANG_CUDA_STUB_API CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit);
TILELANG_CUDA_STUB_API CUresult cuCtxSetLimit(CUlimit limit, size_t value);
TILELANG_CUDA_STUB_API CUresult cuCtxResetPersistingL2Cache(void);
TILELANG_CUDA_STUB_API CUresult cuDeviceGetName(char *name, int len,
                                                CUdevice dev);
TILELANG_CUDA_STUB_API CUresult cuDeviceGetAttribute(int *pi,
                                                     CUdevice_attribute attrib,
                                                     CUdevice dev);
TILELANG_CUDA_STUB_API CUresult cuModuleLoadData(CUmodule *module,
                                                 const void *image);
TILELANG_CUDA_STUB_API CUresult cuModuleLoadDataEx(CUmodule *module,
                                                   const void *image,
                                                   unsigned int numOptions,
                                                   CUjit_option *options,
                                                   void **optionValues);
TILELANG_CUDA_STUB_API CUresult cuModuleUnload(CUmodule hmod);
TILELANG_CUDA_STUB_API CUresult cuModuleGetFunction(CUfunction *hfunc,
                                                    CUmodule hmod,
                                                    const char *name);
TILELANG_CUDA_STUB_API CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr,
                                                     size_t *bytes,
                                                     CUmodule hmod,
                                                     const char *name);
TILELANG_CUDA_STUB_API CUresult cuFuncSetAttribute(CUfunction hfunc,
                                                   CUfunction_attribute attrib,
                                                   int value);
TILELANG_CUDA_STUB_API CUresult cuLaunchKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);
TILELANG_CUDA_STUB_API CUresult cuLaunchKernelEx(const CUlaunchConfig *config,
                                                 CUfunction f,
                                                 void **kernelParams,
                                                 void **extra);
TILELANG_CUDA_STUB_API CUresult cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams);
TILELANG_CUDA_STUB_API CUresult cuMemsetD32_v2(CUdeviceptr dstDevice,
                                               unsigned int ui, size_t N);
TILELANG_CUDA_STUB_API CUresult cuStreamSetAttribute(
    CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value);

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
TILELANG_CUDA_STUB_API CUresult cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);
TILELANG_CUDA_STUB_API CUresult cuTensorMapEncodeIm2col(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
#endif

} // extern "C"
