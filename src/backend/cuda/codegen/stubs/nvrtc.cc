/**
 * \file nvrtc.cc
 * \brief NVRTC stub library for lazy loading libnvrtc.so at runtime.
 *
 * Motivation
 * ----------
 * Similar to cudart, the primary purpose is to resolve SONAME mismatches,
 * allowing a single build to work across different CUDA versions. This is
 * achieved by reusing the NVRTC library already loaded by frameworks like
 * PyTorch.
 *
 * This stub exports a minimal set of NVRTC C API entrypoints used by
 * TVM/TileLang. The actual libnvrtc is loaded lazily via dlopen() on first API
 * call, and symbols are resolved via dlsym().
 */

#include "dynlib.h"
#include <nvrtc.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define TILELANG_NVRTC_STUB_API
#else
#define TILELANG_NVRTC_STUB_API __attribute__((visibility("default")))
#endif

namespace {

#if defined(_WIN32) && !defined(__CYGWIN__)
constexpr const char *kLibNvrtcPaths[] = {
    "nvrtc64_130_0.dll", "nvrtc64_130.dll",   "nvrtc64_13.dll",
    "nvrtc64_120_0.dll", "nvrtc64_120.dll",   "nvrtc64_12.dll",
    "nvrtc64_112_0.dll", "nvrtc64_110_0.dll", "nvrtc64_110.dll",
};
#else
constexpr const char *kLibNvrtcPaths[] = {
    "libnvrtc.so",
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11",
};
#endif

void *TryLoadLibNvrtc() {
#if defined(_WIN32) && !defined(__CYGWIN__)
  for (const char *path : kLibNvrtcPaths) {
    void *handle = tvm::tl::stubs::dynlib_open(path);
    if (handle != nullptr) {
      return handle;
    }
  }
#endif

  // Check if symbols are already available (e.g. loaded by PyTorch).
  void *handle = tvm::tl::stubs::dynlib_find_loaded(
      "nvrtcVersion", reinterpret_cast<void *>(&nvrtcVersion));
  if (handle != nullptr) {
    return handle;
  }

#if !defined(_WIN32) || defined(__CYGWIN__)
  for (const char *path : kLibNvrtcPaths) {
    handle = tvm::tl::stubs::dynlib_open(path);
    if (handle != nullptr) {
      return handle;
    }
  }
#endif

  fprintf(stderr,
          "TileLang Error: nvrtc symbols not found. "
          "Make sure PyTorch with CUDA is installed before using TileLang.\n");
  abort();
}

template <typename T> T GetSymbol(void *handle, const char *name) {
  void *sym = tvm::tl::stubs::dynlib_sym(handle, name);
  const char *error = tvm::tl::stubs::dynlib_error();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

struct NVRTCAPI {
  decltype(&::nvrtcGetErrorString) nvrtcGetErrorString_{nullptr};
  decltype(&::nvrtcVersion) nvrtcVersion_{nullptr};
  decltype(&::nvrtcCreateProgram) nvrtcCreateProgram_{nullptr};
  decltype(&::nvrtcDestroyProgram) nvrtcDestroyProgram_{nullptr};
  decltype(&::nvrtcCompileProgram) nvrtcCompileProgram_{nullptr};
  decltype(&::nvrtcGetPTXSize) nvrtcGetPTXSize_{nullptr};
  decltype(&::nvrtcGetPTX) nvrtcGetPTX_{nullptr};
  decltype(&::nvrtcGetProgramLogSize) nvrtcGetProgramLogSize_{nullptr};
  decltype(&::nvrtcGetProgramLog) nvrtcGetProgramLog_{nullptr};
};

void *GetLibNvrtcHandle() {
  static void *handle = TryLoadLibNvrtc();
  return handle;
}

NVRTCAPI CreateNVRTCAPI() {
  NVRTCAPI api{};
  void *handle = GetLibNvrtcHandle();
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = GetSymbol<decltype(api.name##_)>(handle, #name);               \
  if (api.name##_ == nullptr) {                                                \
    return NVRTCAPI{};                                                         \
  }

  LOOKUP_REQUIRED(nvrtcGetErrorString)
  LOOKUP_REQUIRED(nvrtcVersion)
  LOOKUP_REQUIRED(nvrtcCreateProgram)
  LOOKUP_REQUIRED(nvrtcDestroyProgram)
  LOOKUP_REQUIRED(nvrtcCompileProgram)
  LOOKUP_REQUIRED(nvrtcGetPTXSize)
  LOOKUP_REQUIRED(nvrtcGetPTX)
  LOOKUP_REQUIRED(nvrtcGetProgramLogSize)
  LOOKUP_REQUIRED(nvrtcGetProgramLog)

#undef LOOKUP_REQUIRED

  return api;
}

NVRTCAPI *GetNVRTCAPI() {
  static NVRTCAPI singleton = CreateNVRTCAPI();
  return &singleton;
}

// Provide a stable error string even if libnvrtc cannot be loaded.
const char *FallbackNvrtcErrorString(nvrtcResult result) {
  switch (result) {
  case NVRTC_SUCCESS:
    return "NVRTC_SUCCESS";
  case NVRTC_ERROR_INTERNAL_ERROR:
    return "NVRTC_ERROR_INTERNAL_ERROR (NVRTC stub: libnvrtc not found)";
  default:
    return "NVRTC_ERROR (NVRTC stub: libnvrtc not found)";
  }
}

nvrtcResult MissingLibraryError() { return NVRTC_ERROR_INTERNAL_ERROR; }

} // namespace

extern "C" {

TILELANG_NVRTC_STUB_API const char *nvrtcGetErrorString(nvrtcResult result) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcGetErrorString_ != nullptr) {
    return api->nvrtcGetErrorString_(result);
  }
  return FallbackNvrtcErrorString(result);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcVersion(int *major, int *minor) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcVersion_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcVersion_(major, minor);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcCreateProgram(
    nvrtcProgram *prog, const char *src, const char *name, int numHeaders,
    const char *const *headers, const char *const *includeNames) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcCreateProgram_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcCreateProgram_(prog, src, name, numHeaders, headers,
                                  includeNames);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcDestroyProgram_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcDestroyProgram_(prog);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcCompileProgram(
    nvrtcProgram prog, int numOptions, const char *const *options) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcCompileProgram_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcCompileProgram_(prog, numOptions, options);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog,
                                                    size_t *ptxSizeRet) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcGetPTXSize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcGetPTXSize_(prog, ptxSizeRet);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcGetPTX_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcGetPTX_(prog, ptx);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog,
                                                           size_t *logSizeRet) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcGetProgramLogSize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcGetProgramLogSize_(prog, logSizeRet);
}

TILELANG_NVRTC_STUB_API nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog,
                                                       char *log) {
  auto *api = GetNVRTCAPI();
  if (api->nvrtcGetProgramLog_ == nullptr) {
    return MissingLibraryError();
  }
  return api->nvrtcGetProgramLog_(prog, log);
}

} // extern "C"
