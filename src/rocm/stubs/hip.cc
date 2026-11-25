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
 * - Device-existence and error-string probes (hipGetDeviceCount,
 *   hipGetErrorName, hipGetErrorString) degrade gracefully without throwing so
 *   a ROCm-enabled build can be imported and probed on machines without ROCm.
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

#include <mutex>
#include <stdexcept>
#include <string>

namespace tvm::tl::hip {

namespace {

constexpr const char *kLibHipPaths[] = {
    "libamdhip64.so",
    // Some distros ship a versioned SONAME as well; try a few common ones.
    "libamdhip64.so.7",
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
  // Reuse a runtime that is already mapped (e.g. PyTorch's wheel-bundled copy
  // loaded with RTLD_LOCAL) before loading another one from the system path.
  // Two runtimes in one process make module handles incompatible.
  for (const char *path : kLibHipPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle != nullptr) {
      return handle;
    }
  }
  for (const char *path : kLibHipPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

HIPDriverAPI CreateHIPDriverAPI(void *handle, std::string *missing_symbol) {
  HIPDriverAPI api{};
  if (handle == nullptr) {
    return api;
  }

#define TILELANG_STRINGIFY_IMPL(symbol) #symbol
#define TILELANG_STRINGIFY(symbol) TILELANG_STRINGIFY_IMPL(symbol)
#define LOOKUP(member, symbol)                                                 \
  api.member = GetSymbol<decltype(api.member)>(handle, symbol);                \
  if (api.member == nullptr) {                                                 \
    if (missing_symbol != nullptr) {                                           \
      *missing_symbol = symbol;                                                \
    }                                                                          \
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

struct HIPDriverState {
  std::mutex mu;
  void *handle{nullptr};
  HIPDriverAPI api{};
  std::string missing_symbol;
};

HIPDriverState &GetHIPDriverState() {
  static HIPDriverState state;
  std::lock_guard<std::mutex> lock(state.mu);
  // Retry loading until a handle is found so an early existence probe before
  // importing a ROCm framework (e.g. PyTorch wheel with RTLD_LOCAL libamdhip64)
  // does not permanently cache nullptr.
  if (state.handle == nullptr) {
    state.handle = TryLoadLibAmdHip64();
    if (state.handle != nullptr) {
      state.missing_symbol.clear();
      state.api = CreateHIPDriverAPI(state.handle, &state.missing_symbol);
    }
  }
  return state;
}

HIPDriverAPI *GetHIPDriverAPIOrNull() {
  HIPDriverState &state = GetHIPDriverState();
  if (state.handle == nullptr || !state.missing_symbol.empty()) {
    return nullptr;
  }
  return &state.api;
}

} // namespace

void *HIPDriverAPI::get_handle() { return GetHIPDriverState().handle; }

bool HIPDriverAPI::is_available() {
  HIPDriverState &state = GetHIPDriverState();
  return state.handle != nullptr && state.missing_symbol.empty();
}

HIPDriverAPI *HIPDriverAPI::get() {
  HIPDriverState &state = GetHIPDriverState();
  if (state.handle == nullptr) {
    throw std::runtime_error(
        "HIP runtime library (libamdhip64.so) not found. "
        "Install ROCm (or import a ROCm-enabled framework like PyTorch) before "
        "using TileLang's ROCm backend.");
  }
  if (!state.missing_symbol.empty()) {
    throw std::runtime_error(
        "libamdhip64.so was found but does not provide the required symbol `" +
        state.missing_symbol +
        "`. TileLang's ROCm backend requires a compatible ROCm runtime.");
  }
  return &state.api;
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
  HIPDriverAPI *api = tvm::tl::hip::GetHIPDriverAPIOrNull();
  if (api == nullptr) {
    if (error == hipSuccess) {
      return "hipSuccess";
    }
    return error == hipErrorNoDevice ? "hipErrorNoDevice" : "hipErrorUnknown";
  }
  return api->hipGetErrorName_(error);
}

const char *hipGetErrorString(hipError_t error) {
  HIPDriverAPI *api = tvm::tl::hip::GetHIPDriverAPIOrNull();
  if (api == nullptr) {
    return error == hipSuccess
               ? "hipSuccess"
               : "HIP runtime library (libamdhip64.so) not found";
  }
  return api->hipGetErrorString_(error);
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
  // Device-existence probes (e.g. TVM's ROCMDeviceAPI kExist check) call
  // hipGetDeviceCount and expect a non-hipSuccess status rather than a C++
  // exception when no ROCm runtime is installed.
  if (count != nullptr) {
    *count = 0;
  }
  HIPDriverAPI *api = tvm::tl::hip::GetHIPDriverAPIOrNull();
  if (api == nullptr) {
    return hipErrorNoDevice;
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
