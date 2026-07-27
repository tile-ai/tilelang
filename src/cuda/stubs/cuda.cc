/**
 * \file cuda.cc
 * \brief Implementation of CUDA driver API stub library.
 *
 * This file implements lazy loading of libcuda.so and provides global wrapper
 * functions that serve as drop-in replacements for the CUDA driver API.
 *
 * Motivation
 * ----------
 * The primary purpose is to allow TileLang to be imported on systems without
 * a GPU (e.g., CI/compilation nodes). The library is loaded on first API call
 * using dlopen(). If loading fails, an exception is thrown at call time rather
 * than at import time.
 */

#include "cuda.h"
#include "dynlib.h"

#include <stdexcept>
#include <string>

namespace tvm::tl::cuda {

namespace {

// Library names to try loading (in order of preference)
#if defined(_WIN32) && !defined(__CYGWIN__)
constexpr const char *kLibCudaPaths[] = {
    "nvcuda.dll",
};
#else
constexpr const char *kLibCudaPaths[] = {
    "libcuda.so.1", // Versioned library (most common)
    "libcuda.so",   // Unversioned library
};
#endif

/**
 * \brief Try to load libcuda.so from various paths.
 * \return The dlopen handle, or nullptr if loading failed.
 */
void *try_load_libcuda() {
  void *handle = nullptr;
  for (const char *path : kLibCudaPaths) {
    handle = tvm::tl::stubs::dynlib_open(path);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

/**
 * \brief Get symbol from library handle, returning nullptr on failure.
 *
 * Failure detection is the nullptr return — caller can then call
 * dynlib_error() once to retrieve the loader's error string.
 */
template <typename T> T get_symbol(void *handle, const char *name) {
  return reinterpret_cast<T>(tvm::tl::stubs::dynlib_sym(handle, name));
}

/**
 * \brief Create and initialize the CUDADriverAPI singleton.
 *
 * This function loads libcuda.so and resolves all function symbols.
 * Required symbols that are missing will cause an exception.
 * Optional symbols that are missing will be set to nullptr.
 *
 * \return The initialized CUDADriverAPI instance.
 * \throws std::runtime_error if a required symbol is missing.
 */
CUDADriverAPI create_driver_api() {
  CUDADriverAPI api{};
  void *handle = CUDADriverAPI::get_handle();

  if (handle == nullptr) {
    return api;
  }

// Lookup required symbols - throw if missing
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);                    \
  if (api.name##_ == nullptr) {                                                \
    const char *error = tvm::tl::stubs::dynlib_error();                        \
    throw std::runtime_error(                                                  \
        std::string("Failed to load required CUDA driver symbol: ") + #name +  \
        ". Error: " + (error ? error : "unknown"));                            \
  }
  TILELANG_LIBCUDA_API_REQUIRED(LOOKUP_REQUIRED)
#undef LOOKUP_REQUIRED

// Lookup optional symbols - set to nullptr if missing (no throw)
#define LOOKUP_OPTIONAL(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);
  TILELANG_LIBCUDA_API_OPTIONAL(LOOKUP_OPTIONAL)
#undef LOOKUP_OPTIONAL

  return api;
}

} // namespace

void *CUDADriverAPI::get_handle() {
  // Static handle ensures library is loaded only once
  static void *handle = try_load_libcuda();
  return handle;
}

bool CUDADriverAPI::is_available() { return get_handle() != nullptr; }

CUDADriverAPI *CUDADriverAPI::get() {
  static CUDADriverAPI singleton = create_driver_api();

  if (!is_available()) {
    throw std::runtime_error(
#if defined(_WIN32) && !defined(__CYGWIN__)
        "CUDA driver library (nvcuda.dll) not found. "
#else
        "CUDA driver library (libcuda.so) not found. "
#endif
        "Please ensure NVIDIA drivers are installed, or use CPU-only mode.");
  }

  return &singleton;
}

} // namespace tvm::tl::cuda

// ============================================================================
// Global wrapper function implementations
// ============================================================================
// These are the implementations for the extern "C" functions declared in the
// header. They provide ABI-compatible replacements for libcuda.so functions.

using tvm::tl::cuda::CUDADriverAPI;

extern "C" {

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  return CUDADriverAPI::get()->cuGetErrorName_(error, pStr);
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
  return CUDADriverAPI::get()->cuGetErrorString_(error, pStr);
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  return CUDADriverAPI::get()->cuCtxGetDevice_(device);
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  return CUDADriverAPI::get()->cuCtxGetLimit_(pvalue, limit);
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  return CUDADriverAPI::get()->cuCtxSetLimit_(limit, value);
}

CUresult CUDAAPI cuCtxResetPersistingL2Cache(void) {
  return CUDADriverAPI::get()->cuCtxResetPersistingL2Cache_();
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  return CUDADriverAPI::get()->cuDeviceGetName_(name, len, dev);
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  return CUDADriverAPI::get()->cuDeviceGetAttribute_(pi, attrib, dev);
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  return CUDADriverAPI::get()->cuModuleLoadData_(module, image);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  return CUDADriverAPI::get()->cuModuleLoadDataEx_(module, image, numOptions,
                                                   options, optionValues);
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  return CUDADriverAPI::get()->cuModuleUnload_(hmod);
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  return CUDADriverAPI::get()->cuModuleGetFunction_(hfunc, hmod, name);
}

CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes,
                                      CUmodule hmod, const char *name) {
  return CUDADriverAPI::get()->cuModuleGetGlobal_(dptr, bytes, hmod, name);
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
  return CUDADriverAPI::get()->cuFuncSetAttribute_(hfunc, attrib, value);
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  return CUDADriverAPI::get()->cuLaunchKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, kernelParams, extra);
}

CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                  void **kernelParams, void **extra) {
  return CUDADriverAPI::get()->cuLaunchKernelEx_(config, f, kernelParams,
                                                 extra);
}

CUresult CUDAAPI cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  return CUDADriverAPI::get()->cuLaunchCooperativeKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, kernelParams);
}

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                size_t N) {
  return CUDADriverAPI::get()->cuMemsetD32_(dstDevice, ui, N);
}

CUresult CUDAAPI cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                                      const CUstreamAttrValue *value) {
  return CUDADriverAPI::get()->cuStreamSetAttribute_(hStream, attr, value);
}

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
CUresult CUDAAPI cuTensorMapEncodeTiled(
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

CUresult CUDAAPI cuTensorMapEncodeIm2col(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
  auto fn = CUDADriverAPI::get()->cuTensorMapEncodeIm2col_;
  if (fn == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return fn(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
            globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner,
            channelsPerPixel, pixelsPerColumn, elementStrides, interleave,
            swizzle, l2Promotion, oobFill);
}
#endif

} // extern "C"
