/**
 * \file nvrtc.cc
 * \brief NVRTC stub library for lazy loading libnvrtc.so at runtime.
 *
 * Motivation
 * ----------
 * NVRTC's SONAME encodes its major version (e.g. libnvrtc.so.12,
 * libnvrtc.so.13). If we link libtvm.so directly against a specific SONAME, a
 * wheel built in one CUDA toolkit environment becomes unusable in another
 * environment that only provides a different NVRTC major version.
 *
 * This stub exports a minimal set of NVRTC C API entrypoints used by
 * TVM/TileLang. The actual libnvrtc is loaded lazily via dlopen() on first API
 * call, and symbols are resolved via dlsym().
 *
 * As a result, the final wheel can run in environments that have NVRTC from
 * CUDA 11/12/13 available (as long as the required symbols exist).
 */

#include <nvrtc.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "nvrtc_stub is currently POSIX-only (requires <dlfcn.h> / dlopen). "        \
    "On Windows, build TileLang from source with -DTILELANG_USE_CUDA_STUBS=OFF " \
    "to link against the real CUDA libraries."
#endif

#include <dlfcn.h>
#include <stddef.h>

// Export symbols with default visibility for the shared stub library.
#define TILELANG_NVRTC_STUB_API __attribute__((visibility("default")))

namespace {

// Try multiple major versions for cross-toolkit compatibility.
constexpr const char *kLibNvrtcPaths[] = {
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    // CUDA 11 typically uses `libnvrtc.so.11.2` (and may also provide a
    // `libnvrtc.so.11` symlink depending on the packaging).
    "libnvrtc.so.11.2",
    "libnvrtc.so.11.1",
    "libnvrtc.so.11.0",
    "libnvrtc.so.11",
    // Unversioned name typically only exists with development packages, but try
    // it as a last resort.
    "libnvrtc.so",
};

void *TryLoadLibNvrtc() {
  // If libnvrtc is already loaded in the current process, prefer reusing that
  // instance to avoid loading multiple NVRTC versions in one process.
#ifdef RTLD_NOLOAD
  for (const char *path : kLibNvrtcPaths) {
    void *existing = dlopen(path, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    if (existing != nullptr) {
      return existing;
    }
  }
#endif

  void *handle = nullptr;
  for (const char *path : kLibNvrtcPaths) {
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
  if (handle == nullptr) {
    return api;
  }

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
