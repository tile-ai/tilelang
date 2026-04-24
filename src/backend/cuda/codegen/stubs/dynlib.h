#pragma once

// Cross-platform dynamic library loading abstraction.
// Provides dlopen/dlsym/dlerror equivalents on Windows via
// LoadLibrary/GetProcAddress, and a portable way to search for symbols in
// already-loaded modules (RTLD_DEFAULT equivalent).

#if defined(_WIN32) && !defined(__CYGWIN__)
// ============================================================================
// Windows implementation
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// clang-format off
#include <windows.h>  // must precede psapi.h
#include <psapi.h>
// clang-format on
#include <cstdio>
#include <cstdlib>

namespace tvm::tl::stubs {

inline void *dynlib_open(const char *path) {
  HMODULE handle =
      LoadLibraryExA(path, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (handle == nullptr) {
    handle = LoadLibraryA(path);
  }
  return reinterpret_cast<void *>(handle);
}

inline void *dynlib_sym(void *handle, const char *name) {
  SetLastError(0);
  return reinterpret_cast<void *>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}

inline const char *dynlib_error() {
  thread_local char buf[256];
  DWORD err = GetLastError();
  if (err == 0)
    return nullptr;
  DWORD n = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), nullptr);
  if (n == 0) {
    snprintf(buf, sizeof(buf), "Windows error code %lu",
             static_cast<unsigned long>(err));
  }
  return buf;
}

// Search all loaded modules for a symbol (equivalent to dlsym(RTLD_DEFAULT,
// name)). Returns the module handle containing the symbol, or nullptr.
// |exclude| is the address of the caller's own stub function to avoid
// self-matches.
inline void *dynlib_find_loaded(const char *symbol_name,
                                void *exclude = nullptr) {
  HANDLE process = GetCurrentProcess();
  DWORD buf_bytes = 1024 * sizeof(HMODULE);
  HMODULE *modules = static_cast<HMODULE *>(malloc(buf_bytes));
  if (!modules)
    return nullptr;
  DWORD needed = 0;
  if (!EnumProcessModules(process, modules, buf_bytes, &needed)) {
    free(modules);
    return nullptr;
  }
  if (needed > buf_bytes) {
    buf_bytes = needed;
    HMODULE *resized = static_cast<HMODULE *>(realloc(modules, buf_bytes));
    if (!resized) {
      free(modules);
      return nullptr;
    }
    modules = resized;
    if (!EnumProcessModules(process, modules, buf_bytes, &needed)) {
      free(modules);
      return nullptr;
    }
  }
  DWORD count = needed / sizeof(HMODULE);
  for (DWORD i = 0; i < count; i++) {
    void *sym =
        reinterpret_cast<void *>(GetProcAddress(modules[i], symbol_name));
    if (sym != nullptr && sym != exclude) {
      free(modules);
      return reinterpret_cast<void *>(modules[i]);
    }
  }
  free(modules);
  return nullptr;
}

} // namespace tvm::tl::stubs

#else
// ============================================================================
// POSIX implementation
// ============================================================================

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

namespace tvm::tl::stubs {

inline void *dynlib_open(const char *path) {
  return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}

inline void *dynlib_sym(void *handle, const char *name) {
  (void)dlerror();
  return dlsym(handle, name);
}

inline const char *dynlib_error() { return dlerror(); }

// Search globally for a symbol already loaded by another library (e.g.
// PyTorch). Tries RTLD_DEFAULT first, then RTLD_NEXT. |exclude| is the address
// of the caller's own stub function to avoid self-matches.
inline void *dynlib_find_loaded(const char *symbol_name,
                                void *exclude = nullptr) {
  void *sym = dlsym(RTLD_DEFAULT, symbol_name);
  if (sym != nullptr && sym != exclude) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, symbol_name);
  if (sym != nullptr) {
    return RTLD_NEXT;
  }
  return nullptr;
}

} // namespace tvm::tl::stubs

#endif
