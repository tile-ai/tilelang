/**
 * \file cuda.h
 * \brief Stub library for lazy loading libcuda.so at runtime.
 *
 * This allows tilelang to be imported on CPU-only machines without CUDA.
 *
 * There are two usage patterns:
 *
 * 1. Direct API access (recommended for new code):
 *
 *    ```cpp
 *    #include "target/stubs/cuda.h"
 *    CUresult result =
 *        tvm::tl::cuda::CUDADriverAPI::get()->cuModuleLoadData_(&mod, image);
 *    ```
 *
 * 2. Wrapper macros for existing code (define TILELANG_LAZY_LOAD_LIBCUDA before
 * including):
 *
 *    ```cpp
 *    #define TILELANG_LAZY_LOAD_LIBCUDA
 *    #include "target/stubs/cuda.h"
 *    // Now cuModuleLoadData() calls go through the lazy loader
 *    CUresult result = cuModuleLoadData(&mod, image);
 *    ```
 */

#pragma once

#define TILELANG_LAZY_LOAD_LIBCUDA
#include "origin/cuda.h" // Include the full CUDA driver API types

// X-macro for listing all required CUDA driver API functions.
// Format: _(function_name)
// These are the core functions used by TVM/tilelang CUDA runtime.
#define TILELANG_LIBCUDA_API_REQUIRED(_)                                       \
  _(cuGetErrorName)                                                            \
  _(cuGetErrorString)                                                          \
  _(cuDeviceGetName)                                                           \
  _(cuModuleLoadData)                                                          \
  _(cuModuleLoadDataEx)                                                        \
  _(cuModuleUnload)                                                            \
  _(cuModuleGetFunction)                                                       \
  _(cuModuleGetGlobal)                                                         \
  _(cuFuncSetAttribute)                                                        \
  _(cuLaunchKernel)                                                            \
  _(cuLaunchKernelEx)                                                          \
  _(cuLaunchCooperativeKernel)                                                 \
  _(cuMemsetD32)

// Optional APIs (may not exist in older drivers or specific configurations)
// These are loaded but may be nullptr if not available
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
#define TILELANG_LIBCUDA_API_OPTIONAL(_) _(cuTensorMapEncodeTiled)
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
struct CUDADriverAPI {
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
// Inline wrapper functions for lazy-loaded CUDA driver API
// ============================================================================
// These functions provide a convenient way to call the lazy-loaded API
// without accessing the CUDADriverAPI struct directly.

namespace tvm::tl::cuda {

// Error handling
inline CUresult cuGetErrorName_stub(CUresult error, const char **pStr) {
  return CUDADriverAPI::get()->cuGetErrorName_(error, pStr);
}

inline CUresult cuGetErrorString_stub(CUresult error, const char **pStr) {
  return CUDADriverAPI::get()->cuGetErrorString_(error, pStr);
}

// Device management
inline CUresult cuDeviceGetName_stub(char *name, int len, CUdevice dev) {
  return CUDADriverAPI::get()->cuDeviceGetName_(name, len, dev);
}

// Module management
inline CUresult cuModuleLoadData_stub(CUmodule *module, const void *image) {
  return CUDADriverAPI::get()->cuModuleLoadData_(module, image);
}

inline CUresult cuModuleLoadDataEx_stub(CUmodule *module, const void *image,
                                        unsigned int numOptions,
                                        CUjit_option *options,
                                        void **optionValues) {
  return CUDADriverAPI::get()->cuModuleLoadDataEx_(module, image, numOptions,
                                                   options, optionValues);
}

inline CUresult cuModuleUnload_stub(CUmodule hmod) {
  return CUDADriverAPI::get()->cuModuleUnload_(hmod);
}

inline CUresult cuModuleGetFunction_stub(CUfunction *hfunc, CUmodule hmod,
                                         const char *name) {
  return CUDADriverAPI::get()->cuModuleGetFunction_(hfunc, hmod, name);
}

inline CUresult cuModuleGetGlobal_stub(CUdeviceptr *dptr, size_t *bytes,
                                       CUmodule hmod, const char *name) {
  return CUDADriverAPI::get()->cuModuleGetGlobal_(dptr, bytes, hmod, name);
}

// Function management
inline CUresult cuFuncSetAttribute_stub(CUfunction hfunc,
                                        CUfunction_attribute attrib,
                                        int value) {
  return CUDADriverAPI::get()->cuFuncSetAttribute_(hfunc, attrib, value);
}

// Kernel launching
inline CUresult
cuLaunchKernel_stub(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                    unsigned int gridDimZ, unsigned int blockDimX,
                    unsigned int blockDimY, unsigned int blockDimZ,
                    unsigned int sharedMemBytes, CUstream hStream,
                    void **kernelParams, void **extra) {
  return CUDADriverAPI::get()->cuLaunchKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, kernelParams, extra);
}

inline CUresult cuLaunchKernelEx_stub(const CUlaunchConfig *config,
                                      CUfunction f, void **kernelParams,
                                      void **extra) {
  return CUDADriverAPI::get()->cuLaunchKernelEx_(config, f, kernelParams,
                                                 extra);
}

inline CUresult cuLaunchCooperativeKernel_stub(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  return CUDADriverAPI::get()->cuLaunchCooperativeKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, kernelParams);
}

// Memory operations
inline CUresult cuMemsetD32_stub(CUdeviceptr dstDevice, unsigned int ui,
                                 size_t N) {
  return CUDADriverAPI::get()->cuMemsetD32_(dstDevice, ui, N);
}

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
// TMA (Tensor Memory Access) - CUDA 12.0+
inline CUresult cuTensorMapEncodeTiled_stub(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  auto fn = CUDADriverAPI::get()->cuTensorMapEncodeTiled_;
  if (fn == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return fn(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
            globalStrides, boxDim, elementStrides, interleave, swizzle,
            l2Promotion, oobFill);
}
#endif

} // namespace tvm::tl::cuda

// ============================================================================
// Optional: Macro-based redirection for existing code
// ============================================================================
// When TILELANG_LAZY_LOAD_LIBCUDA is defined, CUDA driver calls are redirected
// to go through the lazy-loading stub. This is useful for adapting existing
// code without modification.
//
// WARNING: This redefines CUDA driver function names as macros, which may
// conflict with other code that uses these names differently. Use with care.

#ifdef TILELANG_LAZY_LOAD_LIBCUDA

// First, undefine the versioned macros from cuda.h
#undef cuModuleGetGlobal
#undef cuMemsetD32
#undef cuLaunchKernel
#undef cuLaunchKernelEx
#undef cuLaunchCooperativeKernel

// Redefine to use our stubs
#define cuGetErrorName ::tvm::tl::cuda::cuGetErrorName_stub
#define cuGetErrorString ::tvm::tl::cuda::cuGetErrorString_stub
#define cuDeviceGetName ::tvm::tl::cuda::cuDeviceGetName_stub
#define cuModuleLoadData ::tvm::tl::cuda::cuModuleLoadData_stub
#define cuModuleLoadDataEx ::tvm::tl::cuda::cuModuleLoadDataEx_stub
#define cuModuleUnload ::tvm::tl::cuda::cuModuleUnload_stub
#define cuModuleGetFunction ::tvm::tl::cuda::cuModuleGetFunction_stub
#define cuModuleGetGlobal ::tvm::tl::cuda::cuModuleGetGlobal_stub
#define cuFuncSetAttribute ::tvm::tl::cuda::cuFuncSetAttribute_stub
#define cuLaunchKernel ::tvm::tl::cuda::cuLaunchKernel_stub
#define cuLaunchKernelEx ::tvm::tl::cuda::cuLaunchKernelEx_stub
#define cuLaunchCooperativeKernel                                              \
  ::tvm::tl::cuda::cuLaunchCooperativeKernel_stub
#define cuMemsetD32 ::tvm::tl::cuda::cuMemsetD32_stub
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
#define cuTensorMapEncodeTiled ::tvm::tl::cuda::cuTensorMapEncodeTiled_stub
#endif

#endif // TILELANG_LAZY_LOAD_LIBCUDA
