/**
 * \file hip.cc
 * \brief Implementation of HIP stub library.
 *
 * This implements lazy loading of libamdhip64.so and provides exported global
 * wrapper functions that serve as drop-in replacements for the HIP runtime /
 * module APIs used by TVM/TileLang.
 *
 * The implementation mirrors src/target/stubs/cuda.cc:
 * - Resolve symbols via dlopen/dlsym on first use.
 * - Prefer RTLD_DEFAULT/RTLD_NEXT when HIP is already loaded by another
 *   framework (e.g. PyTorch ROCm).
 *
 * Additionally, this stub provides wrappers for the minimal HSA APIs used by
 * TVM's ROCm device existence check (hsa_init / hsa_shut_down) so that a ROCm
 * enabled build can still be imported on machines without ROCm installed.
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

// HSA is only used for two entrypoints, and we want to keep this stub
// buildable in environments without ROCm headers installed.
#if __has_include(<hsa/hsa.h>)
#include <hsa/hsa.h>
#define TILELANG_HAS_HSA_HEADERS 1
#else
#define TILELANG_HAS_HSA_HEADERS 0
typedef int hsa_status_t;
#ifndef HSA_STATUS_SUCCESS
#define HSA_STATUS_SUCCESS 0
#endif
extern "C" hsa_status_t hsa_init(void);
extern "C" hsa_status_t hsa_shut_down(void);
#endif

namespace tvm::tl::hip {

namespace {

constexpr const char *kLibHipPaths[] = {
    "libamdhip64.so",
    // Some distros ship a versioned SONAME as well; try a few common ones.
    "libamdhip64.so.6",
    "libamdhip64.so.5",
};

constexpr const char *kLibHsaPaths[] = {
    "libhsa-runtime64.so.1",
    "libhsa-runtime64.so",
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

void *TryLoadLibAmdHip64() {
  // Prefer already-loaded symbols (e.g. if PyTorch ROCm is imported first).
  // We use a representative symbol and ensure we don't just find ourselves.
  void *sym = dlsym(RTLD_DEFAULT, "hipGetErrorString");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&hipGetErrorString)) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, "hipGetErrorString");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&hipGetErrorString)) {
    return RTLD_NEXT;
  }

  // Otherwise, attempt to dlopen the library directly.
  void *handle = nullptr;
  for (const char *path : kLibHipPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

HIPDriverAPI CreateHIPDriverAPI() {
  HIPDriverAPI api{};
  void *handle = HIPDriverAPI::get_handle();
  if (handle == nullptr) {
    return api;
  }

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
  LOOKUP(hipGetDeviceProperties_, "hipGetDeviceProperties")
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
#undef LOOKUP

  return api;
}

// -----------------------------------------------------------------------------
// Minimal HSA stub (needed by TVM's ROCm runtime).
// -----------------------------------------------------------------------------
struct HSAAPI {
  decltype(&::hsa_init) hsa_init_{nullptr};
  decltype(&::hsa_shut_down) hsa_shut_down_{nullptr};
};

void *TryLoadLibHsaRuntime() {
  void *sym = dlsym(RTLD_DEFAULT, "hsa_init");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&hsa_init)) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, "hsa_init");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&hsa_init)) {
    return RTLD_NEXT;
  }

  void *handle = nullptr;
  for (const char *path : kLibHsaPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

void *GetLibHsaHandle() {
  static void *handle = TryLoadLibHsaRuntime();
  return handle;
}

HSAAPI CreateHSAAPI() {
  HSAAPI api{};
  void *handle = GetLibHsaHandle();
  if (handle == nullptr) {
    return api;
  }
  api.hsa_init_ = GetSymbol<decltype(api.hsa_init_)>(handle, "hsa_init");
  api.hsa_shut_down_ =
      GetSymbol<decltype(api.hsa_shut_down_)>(handle, "hsa_shut_down");
  // It's fine if these are nullptr; wrappers will return an error code.
  return api;
}

HSAAPI *GetHSAAPI() {
  static HSAAPI singleton = CreateHSAAPI();
  return &singleton;
}

#if TILELANG_HAS_HSA_HEADERS
static hsa_status_t MissingHsaError() {
  // Any non-success value makes TVM treat ROCm as not existing.
  return static_cast<hsa_status_t>(1);
}
#else
static hsa_status_t MissingHsaError() { return 1; }
#endif

} // namespace

void *HIPDriverAPI::get_handle() {
  static void *handle = TryLoadLibAmdHip64();
  return handle;
}

bool HIPDriverAPI::is_available() { return get_handle() != nullptr; }

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
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return HIPDriverAPI::get()->hipGetDeviceCount_(count);
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

// --- Minimal HSA wrappers
// -------------------------------------------------------

TILELANG_HIP_STUB_API hsa_status_t hsa_init(void) {
  auto *api = tvm::tl::hip::GetHSAAPI();
  if (api->hsa_init_ == nullptr) {
    return tvm::tl::hip::MissingHsaError();
  }
  return api->hsa_init_();
}

TILELANG_HIP_STUB_API hsa_status_t hsa_shut_down(void) {
  auto *api = tvm::tl::hip::GetHSAAPI();
  if (api->hsa_shut_down_ == nullptr) {
    return tvm::tl::hip::MissingHsaError();
  }
  return api->hsa_shut_down_();
}

} // extern "C"
