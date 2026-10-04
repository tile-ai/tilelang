/**
 * \file hip.cc
 * \brief Implementation of HIP stub library.
 *
 * This implements lazy loading of libamdhip64.so and provides exported global
 * wrapper functions that serve as drop-in replacements for the HIP runtime /
 * module APIs used by TVM/TileLang.
 *
 * The implementation mirrors src/cuda/stubs/cuda.cc:
 * - Resolve symbols via dlopen/dlsym on first use.
 * - Prefer RTLD_DEFAULT/RTLD_NEXT when HIP is already loaded by another
 *   framework (e.g. PyTorch ROCm).
 *
 */

#include "hip.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "hip_stub is currently POSIX-only (requires <dlfcn.h> / dlopen). "       \
    "On Windows, build TileLang from source with -DTILELANG_USE_HIP_STUBS=OFF " \
    "to link against the real ROCm libraries."
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include <stdexcept>
#include <string>

namespace tvm::tl::hip {

namespace {

constexpr const char *kLibHipPaths[] = {
    "libamdhip64.so",
    // Some distros ship a versioned SONAME as well; try a few common ones.
    "libamdhip64.so.6",
    "libamdhip64.so.5",
};

template <typename T> T GetSymbol(void *handle, const char *name) {
  (void)dlerror();
  void *sym = dlsym(handle, name);
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

struct HIPLibrary {
  void *handle;
  // RTLD_DEFAULT can be nullptr, so a handle alone cannot indicate
  // availability.
  bool available;
};

HIPLibrary TryLoadLibAmdHip64() {
  // Prefer already-loaded symbols (e.g. if PyTorch ROCm is imported first).
  // We use a representative symbol and ensure we don't just find ourselves.
  void *sym = dlsym(RTLD_DEFAULT, "hipGetErrorString");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&hipGetErrorString)) {
    return {RTLD_DEFAULT, true};
  }
  sym = dlsym(RTLD_NEXT, "hipGetErrorString");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&hipGetErrorString)) {
    return {RTLD_NEXT, true};
  }

  // Otherwise, attempt to dlopen the library directly.
  void *handle = nullptr;
  for (const char *path : kLibHipPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return {handle, handle != nullptr};
}

const HIPLibrary &GetHIPLibrary() {
  static const HIPLibrary library = TryLoadLibAmdHip64();
  return library;
}

HIPDriverAPI CreateHIPDriverAPI() {
  HIPDriverAPI api{};
  void *handle = HIPDriverAPI::get_handle();
  if (!HIPDriverAPI::is_available()) {
    return api;
  }

#define TILELANG_STRINGIFY_IMPL(symbol) #symbol
#define TILELANG_STRINGIFY(symbol) TILELANG_STRINGIFY_IMPL(symbol)
#define LOOKUP(member, symbol)                                                 \
  api.member = GetSymbol<decltype(api.member)>(handle, symbol);                \
  if (api.member == nullptr) {                                                 \
    return HIPDriverAPI{};                                                     \
  }

  LOOKUP(hipGetErrorName_, "hipGetErrorName")
  LOOKUP(hipGetErrorString_, "hipGetErrorString")
  LOOKUP(hipGetLastError_, "hipGetLastError")
  LOOKUP(hipSetDevice_, "hipSetDevice")
  LOOKUP(hipGetDevice_, "hipGetDevice")
  LOOKUP(hipGetDeviceCount_, "hipGetDeviceCount")
  LOOKUP(hipDeviceGetAttribute_, "hipDeviceGetAttribute")
  LOOKUP(hipDeviceGetName_, "hipDeviceGetName")
  // ROCm 6+ maps this API to an ABI-versioned symbol such as
  // hipGetDevicePropertiesR0600. Resolve the symbol selected by the build
  // headers instead of casting the legacy entrypoint to the new struct type.
  LOOKUP(hipGetDeviceProperties_, TILELANG_STRINGIFY(hipGetDeviceProperties))
  LOOKUP(hipMalloc_, "hipMalloc")
  LOOKUP(hipFree_, "hipFree")
  LOOKUP(hipHostMalloc_, "hipHostMalloc")
  LOOKUP(hipHostFree_, "hipHostFree")
  LOOKUP(hipMemcpy_, "hipMemcpy")
  LOOKUP(hipMemcpyAsync_, "hipMemcpyAsync")
  LOOKUP(hipMemcpyPeerAsync_, "hipMemcpyPeerAsync")
  LOOKUP(hipStreamCreate_, "hipStreamCreate")
  LOOKUP(hipStreamDestroy_, "hipStreamDestroy")
  LOOKUP(hipStreamSynchronize_, "hipStreamSynchronize")
  LOOKUP(hipEventCreate_, "hipEventCreate")
  LOOKUP(hipEventDestroy_, "hipEventDestroy")
  LOOKUP(hipEventRecord_, "hipEventRecord")
  LOOKUP(hipEventSynchronize_, "hipEventSynchronize")
  LOOKUP(hipEventElapsedTime_, "hipEventElapsedTime")
  LOOKUP(hipModuleLoadData_, "hipModuleLoadData")
  LOOKUP(hipModuleUnload_, "hipModuleUnload")
  LOOKUP(hipModuleGetFunction_, "hipModuleGetFunction")
  LOOKUP(hipModuleGetGlobal_, "hipModuleGetGlobal")
  LOOKUP(hipModuleLaunchKernel_, "hipModuleLaunchKernel")
  LOOKUP(hipModuleLaunchCooperativeKernel_, "hipModuleLaunchCooperativeKernel")
#undef LOOKUP
#undef TILELANG_STRINGIFY
#undef TILELANG_STRINGIFY_IMPL

  return api;
}

} // namespace

void *HIPDriverAPI::get_handle() { return GetHIPLibrary().handle; }

bool HIPDriverAPI::is_available() { return GetHIPLibrary().available; }

HIPDriverAPI *HIPDriverAPI::get() {
  static HIPDriverAPI singleton = CreateHIPDriverAPI();
  if (!is_available()) {
    throw std::runtime_error(
        "HIP runtime library (libamdhip64.so) not found. "
        "Install ROCm (or import a ROCm-enabled framework like PyTorch) before "
        "using TileLang's ROCm backend.");
  }
  return &singleton;
}

} // namespace tvm::tl::hip

// ============================================================================
// Global wrapper function implementations
// ============================================================================

using tvm::tl::hip::HIPDriverAPI;

extern "C" {

// --- HIP runtime/module wrappers
// ------------------------------------------------

const char *hipGetErrorName(hipError_t error) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipGetErrorName_(error);
}

const char *hipGetErrorString(hipError_t error) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipGetErrorString_(error);
}

hipError_t hipGetLastError(void) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipGetLastError_();
}

hipError_t hipSetDevice(int deviceId) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipSetDevice_(deviceId);
}

hipError_t hipGetDevice(int *deviceId) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipGetDevice_(deviceId);
}

hipError_t hipGetDeviceCount(int *count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = 0;
  // Device existence queries must work without a HIP runtime installed.
  if (!HIPDriverAPI::is_available()) {
    return hipErrorSharedObjectInitFailed;
  }
  auto *api = HIPDriverAPI::get();
  if (api->hipGetDeviceCount_ == nullptr) {
    return hipErrorSharedObjectSymbolNotFound;
  }
  return api->hipGetDeviceCount_(count);
}

hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipDeviceGetAttribute_(pi, attr, deviceId);
}

hipError_t hipDeviceGetName(char *name, int len, int deviceId) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipDeviceGetName_(name, len, deviceId);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipGetDeviceProperties_(prop, deviceId);
}

hipError_t hipMalloc(void **ptr, size_t size) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipMalloc_(ptr, size);
}

// NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
hipError_t hipFree(void *ptr) { return HIPDriverAPI::get()->hipFree_(ptr); }

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipHostMalloc_(ptr, size, flags);
}

hipError_t hipHostFree(void *ptr) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipHostFree_(ptr);
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipMemcpy_(dst, src, sizeBytes, kind);
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipMemcpyAsync_(dst, src, sizeBytes, kind,
                                              stream);
}

hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src,
                              int srcDeviceId, size_t sizeBytes,
                              hipStream_t stream) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipMemcpyPeerAsync_(
      dst, dstDeviceId, src, srcDeviceId, sizeBytes, stream);
}

hipError_t hipStreamCreate(hipStream_t *stream) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipStreamCreate_(stream);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipStreamDestroy_(stream);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipStreamSynchronize_(stream);
}

hipError_t hipEventCreate(hipEvent_t *event) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipEventCreate_(event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipEventDestroy_(event);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipEventRecord_(event, stream);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipEventSynchronize_(event);
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipEventElapsedTime_(ms, start, stop);
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipModuleLoadData_(module, image);
}

hipError_t hipModuleUnload(hipModule_t module) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipModuleUnload_(module);
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module,
                                const char *name) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipModuleGetFunction_(function, module, name);
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t module, const char *name) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipModuleGetGlobal_(dptr, bytes, module, name);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams,
                                 void **extra) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipModuleLaunchKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);
}

hipError_t hipModuleLaunchCooperativeKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void **kernelParams) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipModuleLaunchCooperativeKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams);
}

} // extern "C"
