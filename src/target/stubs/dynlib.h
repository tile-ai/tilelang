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

namespace tvm::tl::stubs {

inline void *dynlib_open(const char *path) {
  return reinterpret_cast<void *>(LoadLibraryA(path));
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
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                 nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf,
                 sizeof(buf), nullptr);
  return buf;
}

// Search all loaded modules for a symbol (equivalent to dlsym(RTLD_DEFAULT,
// name)). Returns the module handle containing the symbol, or nullptr.
// |exclude| is the address of the caller's own stub function to avoid
// self-matches.
inline void *dynlib_find_loaded(const char *symbol_name,
                                void *exclude = nullptr) {
  HMODULE modules[1024];
  DWORD needed = 0;
  if (!EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules),
                          &needed)) {
    return nullptr;
  }
  DWORD count = needed / sizeof(HMODULE);
  for (DWORD i = 0; i < count; i++) {
    void *sym =
        reinterpret_cast<void *>(GetProcAddress(modules[i], symbol_name));
    if (sym != nullptr && sym != exclude) {
      return reinterpret_cast<void *>(modules[i]);
    }
  }
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
