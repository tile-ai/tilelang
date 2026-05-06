/**
 * \file hiprtc.cc
 * \brief HIPRTC stub library for lazy loading libhiprtc.so at runtime.
 *
 * Similar to src/target/stubs/nvrtc.cc, this stub exports a minimal subset of
 * the HIPRTC C API and resolves the real implementation with dlopen()/dlsym().
 *
 * This allows a ROCm-enabled TileLang build to be imported on machines without
 * ROCm installed, and avoids hard DT_NEEDED dependencies on libhiprtc.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#if __has_include(<hip/hiprtc.h>)
#include <hip/hiprtc.h>
#define TILELANG_HAS_HIPRTC_HEADERS 1
#else
#define TILELANG_HAS_HIPRTC_HEADERS 0
// Minimal fallback definitions to keep this file buildable without ROCm SDK.
// These are intentionally incomplete; prefer building with real ROCm headers.
#include <stddef.h>
typedef struct _hiprtcProgram *hiprtcProgram;
typedef enum hiprtcResult {
  HIPRTC_SUCCESS = 0,
  HIPRTC_ERROR_INTERNAL_ERROR = 1
} hiprtcResult;

#ifdef __cplusplus
extern "C" {
#endif
const char *hiprtcGetErrorString(hiprtcResult result);
hiprtcResult hiprtcVersion(int *major, int *minor);
hiprtcResult hiprtcCreateProgram(hiprtcProgram *prog, const char *src,
                                 const char *name, int numHeaders,
                                 const char *const *headers,
                                 const char *const *includeNames);
hiprtcResult hiprtcDestroyProgram(hiprtcProgram *prog);
hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions,
                                  const char *const *options);
hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t *codeSizeRet);
hiprtcResult hiprtcGetCode(hiprtcProgram prog, char *code);
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t *logSizeRet);
hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char *log);
#ifdef __cplusplus
} // extern "C"
#endif
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "hiprtc_stub is currently POSIX-only (requires <dlfcn.h> / dlopen). "     \
    "On Windows, build TileLang from source with -DTILELANG_USE_HIP_STUBS=OFF " \
    "to link against the real ROCm libraries."
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Export symbols with default visibility for the shared stub library.
#define TILELANG_HIPRTC_STUB_API __attribute__((visibility("default")))

namespace {

constexpr const char *kLibHiprtcPaths[] = {
    "libhiprtc.so",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
};

void *TryLoadLibHiprtc() {
  // Reuse already-loaded HIPRTC if present (e.g. framework import order).
  void *sym = dlsym(RTLD_DEFAULT, "hiprtcGetErrorString");
  if (sym != nullptr &&
      sym != reinterpret_cast<void *>(&hiprtcGetErrorString)) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, "hiprtcGetErrorString");
  if (sym != nullptr &&
      sym != reinterpret_cast<void *>(&hiprtcGetErrorString)) {
    return RTLD_NEXT;
  }

  void *handle = nullptr;
  for (const char *path : kLibHiprtcPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

template <typename T> T GetSymbol(void *handle, const char *name) {
  (void)dlerror();
  void *sym = dlsym(handle, name);
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

struct HIPRTCAPI {
  decltype(&::hiprtcGetErrorString) hiprtcGetErrorString_{nullptr};
  decltype(&::hiprtcVersion) hiprtcVersion_{nullptr};
  decltype(&::hiprtcCreateProgram) hiprtcCreateProgram_{nullptr};
  decltype(&::hiprtcDestroyProgram) hiprtcDestroyProgram_{nullptr};
  decltype(&::hiprtcCompileProgram) hiprtcCompileProgram_{nullptr};
  decltype(&::hiprtcGetCodeSize) hiprtcGetCodeSize_{nullptr};
  decltype(&::hiprtcGetCode) hiprtcGetCode_{nullptr};
  decltype(&::hiprtcGetProgramLogSize) hiprtcGetProgramLogSize_{nullptr};
  decltype(&::hiprtcGetProgramLog) hiprtcGetProgramLog_{nullptr};
};

void *GetLibHiprtcHandle() {
  static void *handle = TryLoadLibHiprtc();
  return handle;
}

HIPRTCAPI CreateHIPRTCAPI() {
  HIPRTCAPI api{};
  void *handle = GetLibHiprtcHandle();
  if (handle == nullptr) {
    return api;
  }

#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = GetSymbol<decltype(api.name##_)>(handle, #name);               \
  if (api.name##_ == nullptr) {                                                \
    return HIPRTCAPI{};                                                        \
  }

  // hiprtcGetErrorString is optional; we can provide a fallback string.
  api.hiprtcGetErrorString_ = GetSymbol<decltype(api.hiprtcGetErrorString_)>(
      handle, "hiprtcGetErrorString");

  LOOKUP_REQUIRED(hiprtcVersion)
  LOOKUP_REQUIRED(hiprtcCreateProgram)
  LOOKUP_REQUIRED(hiprtcDestroyProgram)
  LOOKUP_REQUIRED(hiprtcCompileProgram)
  LOOKUP_REQUIRED(hiprtcGetCodeSize)
  LOOKUP_REQUIRED(hiprtcGetCode)
  LOOKUP_REQUIRED(hiprtcGetProgramLogSize)
  LOOKUP_REQUIRED(hiprtcGetProgramLog)

#undef LOOKUP_REQUIRED

  return api;
}

HIPRTCAPI *GetHIPRTCAPI() {
  static HIPRTCAPI singleton = CreateHIPRTCAPI();
  return &singleton;
}

hiprtcResult MissingLibraryError() {
#if TILELANG_HAS_HIPRTC_HEADERS
  return HIPRTC_ERROR_INTERNAL_ERROR;
#else
  return HIPRTC_ERROR_INTERNAL_ERROR;
#endif
}

const char *FallbackHiprtcErrorString(hiprtcResult result) {
  switch (result) {
  case HIPRTC_SUCCESS:
    return "HIPRTC_SUCCESS";
  default:
    return "HIPRTC_ERROR (HIPRTC stub: libhiprtc not found)";
  }
}

} // namespace

extern "C" {

TILELANG_HIPRTC_STUB_API const char *hiprtcGetErrorString(hiprtcResult result) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcGetErrorString_ != nullptr) {
    return api->hiprtcGetErrorString_(result);
  }
  return FallbackHiprtcErrorString(result);
}

TILELANG_HIPRTC_STUB_API hiprtcResult hiprtcVersion(int *major, int *minor) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcVersion_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcVersion_(major, minor);
}

TILELANG_HIPRTC_STUB_API hiprtcResult hiprtcCreateProgram(
    hiprtcProgram *prog, const char *src, const char *name, int numHeaders,
    const char *const *headers, const char *const *includeNames) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcCreateProgram_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcCreateProgram_(prog, src, name, numHeaders, headers,
                                   includeNames);
}

TILELANG_HIPRTC_STUB_API hiprtcResult
hiprtcDestroyProgram(hiprtcProgram *prog) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcDestroyProgram_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcDestroyProgram_(prog);
}

TILELANG_HIPRTC_STUB_API hiprtcResult hiprtcCompileProgram(
    hiprtcProgram prog, int numOptions, const char *const *options) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcCompileProgram_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcCompileProgram_(prog, numOptions, options);
}

TILELANG_HIPRTC_STUB_API hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog,
                                                        size_t *codeSizeRet) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcGetCodeSize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcGetCodeSize_(prog, codeSizeRet);
}

TILELANG_HIPRTC_STUB_API hiprtcResult hiprtcGetCode(hiprtcProgram prog,
                                                    char *code) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcGetCode_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcGetCode_(prog, code);
}

TILELANG_HIPRTC_STUB_API hiprtcResult
hiprtcGetProgramLogSize(hiprtcProgram prog, size_t *logSizeRet) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcGetProgramLogSize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcGetProgramLogSize_(prog, logSizeRet);
}

TILELANG_HIPRTC_STUB_API hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog,
                                                          char *log) {
  auto *api = GetHIPRTCAPI();
  if (api->hiprtcGetProgramLog_ == nullptr) {
    return MissingLibraryError();
  }
  return api->hiprtcGetProgramLog_(prog, log);
}

} // extern "C"
