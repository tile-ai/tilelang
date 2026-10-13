/**
 * \file ascendcl.cc
 * \brief Implementation of the CANN runtime (libascendcl) stub library.
 *
 * This implements lazy loading of libascendcl.so and provides exported global
 * wrapper functions that serve as drop-in replacements for the ACL runtime
 * entrypoints used by TileLang's Ascend runtime module.
 *
 * The implementation mirrors src/rocm/stubs/hip.cc:
 * - Resolve symbols via dlopen/dlsym on first use.
 * - Prefer RTLD_DEFAULT/RTLD_NEXT when CANN is already loaded by another
 *   framework (e.g. torch_npu).
 * - Fall back to well-known CANN install roots when LD_LIBRARY_PATH is not
 *   set up (no set_env.sh sourced).
 */

#include "ascendcl.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "ascendcl_stub is POSIX-only (requires <dlfcn.h> / dlopen). "            \
    "The Ascend backend does not support Windows; configure with "             \
    "-DTILELANG_USE_ASCEND_STUBS=OFF."
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace tvm::tl::ascendcl {

namespace {

// First missing required symbol, recorded so get() can name it.
std::string missing_symbol;

template <typename T> T GetSymbol(void *handle, const char *name) {
  (void)dlerror();
  void *sym = dlsym(handle, name);
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

std::vector<std::string> LibAscendCLPathCandidates() {
  std::vector<std::string> candidates;
  // Honors LD_LIBRARY_PATH (set by CANN's set_env.sh).
  candidates.emplace_back("libascendcl.so");

  std::vector<std::string> roots;
  for (const char *env : {"ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"}) {
    const char *value = std::getenv(env);
    if (value != nullptr && value[0] != '\0') {
      roots.emplace_back(value);
    }
  }
  roots.emplace_back("/usr/local/Ascend/ascend-toolkit/latest");

  for (const std::string &root : roots) {
    candidates.push_back(root + "/lib64/libascendcl.so");
    candidates.push_back(root + "/runtime/lib64/libascendcl.so");
  }
  return candidates;
}

void *TryLoadLibAscendCL() {
  // Prefer already-loaded symbols (e.g. if torch_npu is imported first).
  // We use a representative symbol and ensure we don't just find ourselves.
  void *sym = dlsym(RTLD_DEFAULT, "aclrtGetDevice");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&aclrtGetDevice)) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, "aclrtGetDevice");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&aclrtGetDevice)) {
    return RTLD_NEXT;
  }

  // Otherwise, attempt to dlopen the library directly. If a copy is already
  // mapped (e.g. under a different path), the dynamic loader dedups by SONAME
  // and returns the existing mapping.
  for (const std::string &path : LibAscendCLPathCandidates()) {
    void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      return handle;
    }
  }
  return nullptr;
}

AscendCLAPI CreateAscendCLAPI() {
  AscendCLAPI api{};
  void *handle = AscendCLAPI::get_handle();
  if (handle == nullptr) {
    return api;
  }

// Required symbols: record the first missing name (older CANN releases may
// lack e.g. aclrtLaunchKernelWithHostArgs) so get() can report it instead of
// leaving a null pointer behind.
#define LOOKUP(symbol)                                                         \
  api.symbol##_ = GetSymbol<decltype(api.symbol##_)>(handle, #symbol);         \
  if (api.symbol##_ == nullptr) {                                              \
    if (missing_symbol.empty()) {                                              \
      missing_symbol = #symbol;                                                \
    }                                                                          \
    return AscendCLAPI{};                                                      \
  }

  LOOKUP(aclrtBinaryLoadFromData)
  LOOKUP(aclrtBinaryGetFunction)
  LOOKUP(aclrtBinaryUnLoad)
  LOOKUP(aclrtGetDevice)
  LOOKUP(aclrtLaunchKernelWithHostArgs)
#undef LOOKUP

  // Optional: wrappers return nullptr when absent.
  api.aclGetRecentErrMsg_ = GetSymbol<decltype(api.aclGetRecentErrMsg_)>(
      handle, "aclGetRecentErrMsg");
  return api;
}

} // namespace

void *AscendCLAPI::get_handle() {
  static void *handle = TryLoadLibAscendCL();
  return handle;
}

bool AscendCLAPI::is_available() { return get_handle() != nullptr; }

AscendCLAPI *AscendCLAPI::get_or_null() {
  static AscendCLAPI singleton = CreateAscendCLAPI();
  return &singleton;
}

AscendCLAPI *AscendCLAPI::get() {
  AscendCLAPI *api = get_or_null();
  if (!is_available()) {
    throw std::runtime_error(
        "CANN runtime library (libascendcl.so) not found. Install the Ascend "
        "CANN toolkit and source its set_env.sh (or import torch_npu first) "
        "before using TileLang's Ascend backend.");
  }
  if (!missing_symbol.empty()) {
    throw std::runtime_error(
        "libascendcl.so was found but does not provide the required symbol `" +
        missing_symbol +
        "`. TileLang's Ascend backend requires a newer CANN toolkit.");
  }
  return api;
}

} // namespace tvm::tl::ascendcl

// ============================================================================
// Global wrapper function implementations
// ============================================================================

using tvm::tl::ascendcl::AscendCLAPI;

extern "C" {

int32_t aclrtBinaryLoadFromData(const void *data, size_t size,
                                const void *options, void **binHandle) {
  return AscendCLAPI::get()->aclrtBinaryLoadFromData_(data, size, options,
                                                      binHandle);
}

int32_t aclrtBinaryGetFunction(void *binHandle, const char *name,
                               void **funcHandle) {
  return AscendCLAPI::get()->aclrtBinaryGetFunction_(binHandle, name,
                                                     funcHandle);
}

int32_t aclrtBinaryUnLoad(void *binHandle) {
  return AscendCLAPI::get()->aclrtBinaryUnLoad_(binHandle);
}

int32_t aclrtGetDevice(int32_t *deviceId) {
  return AscendCLAPI::get()->aclrtGetDevice_(deviceId);
}

int32_t aclrtLaunchKernelWithHostArgs(void *func, uint32_t numBlocks,
                                      void *stream, void *config, void *args,
                                      size_t argsSize, void *reserved,
                                      size_t reservedSize) {
  return AscendCLAPI::get()->aclrtLaunchKernelWithHostArgs_(
      func, numBlocks, stream, config, args, argsSize, reserved, reservedSize);
}

const char *aclGetRecentErrMsg(void) {
  // Error-message path: must never throw, callers handle nullptr.
  AscendCLAPI *api = AscendCLAPI::get_or_null();
  if (!AscendCLAPI::is_available() || api->aclGetRecentErrMsg_ == nullptr) {
    return nullptr;
  }
  return api->aclGetRecentErrMsg_();
}

} // extern "C"
