#pragma once

// Cross-platform dynamic library loading abstraction.
// Provides dlopen/dlsym/dlerror equivalents on Windows via
// LoadLibrary/GetProcAddress, and a portable way to search for symbols in
// already-loaded modules (RTLD_DEFAULT equivalent).

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32) && !defined(__CYGWIN__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// clang-format off
#include <windows.h>  // must precede psapi.h
#include <psapi.h>
// clang-format on
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <glob.h>
#include <limits.h>
#include <link.h>
#endif
#endif

namespace tvm::tl::stubs {

#if defined(_WIN32) && !defined(__CYGWIN__)
constexpr char kDynlibPathSeparator = '\\';
constexpr char kDynlibPathListSeparator = ';';
#else
constexpr char kDynlibPathSeparator = '/';
constexpr char kDynlibPathListSeparator = ':';
#endif

inline std::string dynlib_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  });
  return value;
}

inline bool dynlib_starts_with_ci(const std::string &value,
                                  const char *prefix) {
  std::string lower_value = dynlib_lower(value);
  std::string lower_prefix = dynlib_lower(prefix != nullptr ? prefix : "");
  return lower_value.rfind(lower_prefix, 0) == 0;
}

inline bool dynlib_has_path_separator(const char *path) {
  return path != nullptr && (std::strchr(path, '\\') != nullptr ||
                             std::strchr(path, '/') != nullptr);
}

inline std::string dynlib_basename(const std::string &path) {
  size_t pos = path.find_last_of("\\/");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

inline std::string dynlib_dirname(const std::string &path) {
  size_t pos = path.find_last_of("\\/");
  return pos == std::string::npos ? std::string() : path.substr(0, pos);
}

inline std::string dynlib_join(const std::string &dir,
                               const std::string &name) {
  if (dir.empty()) {
    return name;
  }
  if (name.empty()) {
    return dir;
  }
  if (dir.back() == '\\' || dir.back() == '/') {
    return dir + name;
  }
  return dir + kDynlibPathSeparator + name;
}

inline std::string dynlib_normalize_for_compare(std::string value) {
  while (!value.empty() && (value.back() == '\\' || value.back() == '/')) {
    value.pop_back();
  }
#if defined(_WIN32) && !defined(__CYGWIN__)
  std::replace(value.begin(), value.end(), '/', '\\');
  value = dynlib_lower(value);
#endif
  return value;
}

inline void dynlib_append_unique(std::vector<std::string> *values,
                                 std::string value) {
  value = dynlib_normalize_for_compare(std::move(value));
  if (value.empty()) {
    return;
  }
  for (const std::string &existing : *values) {
    if (dynlib_normalize_for_compare(existing) == value) {
      return;
    }
  }
  values->push_back(std::move(value));
}

inline std::string dynlib_get_env(const char *name) {
#if defined(_WIN32) && !defined(__CYGWIN__)
  DWORD needed = GetEnvironmentVariableA(name, nullptr, 0);
  if (needed == 0) {
    return std::string();
  }
  std::string value(needed, '\0');
  DWORD written = GetEnvironmentVariableA(name, value.data(), needed);
  if (written == 0) {
    return std::string();
  }
  value.resize(written);
  return value;
#else
  const char *value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
#endif
}

#if defined(_WIN32) && !defined(__CYGWIN__)
inline bool dynlib_is_dir(const std::string &path) {
  DWORD attrs = GetFileAttributesA(path.c_str());
  return attrs != INVALID_FILE_ATTRIBUTES &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

inline bool dynlib_is_file(const std::string &path) {
  DWORD attrs = GetFileAttributesA(path.c_str());
  return attrs != INVALID_FILE_ATTRIBUTES &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

inline std::vector<std::string> dynlib_find_paths(const std::string &pattern) {
  std::vector<std::string> result;
  WIN32_FIND_DATAA data;
  HANDLE find = FindFirstFileA(pattern.c_str(), &data);
  if (find == INVALID_HANDLE_VALUE) {
    return result;
  }

  std::string dir = dynlib_dirname(pattern);
  do {
    if (std::strcmp(data.cFileName, ".") == 0 ||
        std::strcmp(data.cFileName, "..") == 0) {
      continue;
    }
    result.push_back(dynlib_join(dir, data.cFileName));
  } while (FindNextFileA(find, &data));
  FindClose(find);
  return result;
}

inline std::string dynlib_module_path(HMODULE module) {
  std::string path(MAX_PATH, '\0');
  for (;;) {
    DWORD written = GetModuleFileNameA(module, path.data(),
                                       static_cast<DWORD>(path.size()));
    if (written == 0) {
      return std::string();
    }
    if (written < path.size() - 1) {
      path.resize(written);
      return path;
    }
    path.resize(path.size() * 2);
  }
}

inline std::string dynlib_exe_path() { return dynlib_module_path(nullptr); }

inline std::string dynlib_module_path_from_address(const void *address) {
  HMODULE module = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(address), &module)) {
    return dynlib_module_path(module);
  }
  return std::string();
}

inline void *dynlib_open(const char *path) {
  DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  if (dynlib_has_path_separator(path)) {
    flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
  }
  HMODULE handle = LoadLibraryExA(path, nullptr, flags);
  if (handle == nullptr && GetLastError() == ERROR_INVALID_PARAMETER) {
    // Legacy fallback for systems without LOAD_LIBRARY_SEARCH_* support.
    handle = LoadLibraryA(path);
  }
  return reinterpret_cast<void *>(handle);
}

inline void dynlib_close(void *handle) {
  FreeLibrary(reinterpret_cast<HMODULE>(handle));
}

inline void *dynlib_sym(void *handle, const char *name) {
  // Clear-before-call mirrors the POSIX dlerror() pattern, so a follow-up
  // dynlib_error() observes only the GetProcAddress failure (if any).
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

inline void *dynlib_loaded_module_with_symbol(HMODULE module,
                                              const char *symbol_name,
                                              void *exclude = nullptr) {
  void *sym = reinterpret_cast<void *>(GetProcAddress(module, symbol_name));
  if (sym != nullptr && sym != exclude) {
    return reinterpret_cast<void *>(module);
  }
  return nullptr;
}

inline bool dynlib_enum_process_modules(std::vector<HMODULE> *modules) {
  HANDLE process = GetCurrentProcess();
  modules->assign(1024, nullptr);
  DWORD needed = 0;
  // Up to 4 retries handles the rare race where a module is loaded between
  // EnumProcessModules calls. In practice this converges on the first call.
  for (int retry = 0; retry < 4; ++retry) {
    DWORD bytes = static_cast<DWORD>(modules->size() * sizeof(HMODULE));
    if (!EnumProcessModules(process, modules->data(), bytes, &needed)) {
      return false;
    }
    if (needed <= bytes) {
      modules->resize(needed / sizeof(HMODULE));
      return true;
    }
    modules->resize(needed / sizeof(HMODULE));
  }
  modules->resize(needed / sizeof(HMODULE));
  return true;
}

// Search all loaded modules for a symbol (equivalent to dlsym(RTLD_DEFAULT,
// name)). Returns the module handle containing the symbol, or nullptr.
// |exclude| is the address of the caller's own stub function to avoid
// self-matches.
inline void *dynlib_find_loaded(const char *symbol_name,
                                void *exclude = nullptr) {
  std::vector<HMODULE> modules;
  if (!dynlib_enum_process_modules(&modules)) {
    return nullptr;
  }
  for (HMODULE mod : modules) {
    void *handle = dynlib_loaded_module_with_symbol(mod, symbol_name, exclude);
    if (handle != nullptr) {
      return handle;
    }
  }
  return nullptr;
}

inline std::vector<std::pair<std::string, void *>>
dynlib_loaded_modules_with_symbol_prefix(const char *prefix,
                                         const char *symbol_name,
                                         void *exclude = nullptr) {
  std::vector<std::pair<std::string, void *>> matches;
  std::vector<HMODULE> modules;
  if (!dynlib_enum_process_modules(&modules)) {
    return matches;
  }

  for (HMODULE mod : modules) {
    std::string path = dynlib_module_path(mod);
    if (!dynlib_starts_with_ci(dynlib_basename(path), prefix)) {
      continue;
    }
    void *handle = dynlib_loaded_module_with_symbol(mod, symbol_name, exclude);
    if (handle != nullptr) {
      matches.emplace_back(std::move(path), handle);
    }
  }
  return matches;
}

inline void dynlib_release_loaded_probe_handle(void *, void *) {}

inline void
dynlib_append_program_files_cuda_dirs(std::vector<std::string> *dirs);

#else

inline bool dynlib_is_dir(const std::string &path) {
  struct stat st;
  return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline bool dynlib_is_file(const std::string &path) {
  struct stat st;
  return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline std::vector<std::string> dynlib_find_paths(const std::string &pattern) {
  std::vector<std::string> paths;
#if defined(__linux__)
  glob_t result;
  std::memset(&result, 0, sizeof(result));
  if (glob(pattern.c_str(), 0, nullptr, &result) == 0) {
    for (size_t i = 0; i < result.gl_pathc; ++i) {
      paths.emplace_back(result.gl_pathv[i]);
    }
  }
  globfree(&result);
#endif
  return paths;
}

inline std::string dynlib_exe_path() {
#if defined(__linux__)
  std::string path(PATH_MAX, '\0');
  ssize_t n = readlink("/proc/self/exe", path.data(), path.size() - 1);
  if (n <= 0) {
    return std::string();
  }
  path.resize(static_cast<size_t>(n));
  return path;
#else
  return std::string();
#endif
}

inline std::string dynlib_module_path_from_address(const void *address) {
  Dl_info info;
  if (dladdr(address, &info) != 0 && info.dli_fname != nullptr) {
    return info.dli_fname;
  }
  return std::string();
}

inline void *dynlib_open(const char *path) {
  return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}

inline void dynlib_close(void *handle) { dlclose(handle); }

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

inline void *dynlib_loaded_module_with_symbol(const std::string &path,
                                              const char *symbol_name,
                                              void *exclude = nullptr) {
#if defined(__linux__) && defined(RTLD_NOLOAD)
  void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (handle == nullptr) {
    return nullptr;
  }
  void *sym = dlsym(handle, symbol_name);
  if (sym != nullptr && sym != exclude) {
    return handle;
  }
  dlclose(handle);
#endif
  return nullptr;
}

#if defined(__linux__)
struct DynlibLoadedSearch {
  const char *prefix{nullptr};
  const char *symbol_name{nullptr};
  void *exclude{nullptr};
  std::vector<std::pair<std::string, void *>> matches;
};

inline int dynlib_collect_loaded_module(struct dl_phdr_info *info, size_t,
                                        void *data) {
  auto *search = reinterpret_cast<DynlibLoadedSearch *>(data);
  if (info->dlpi_name == nullptr || info->dlpi_name[0] == '\0') {
    return 0;
  }
  std::string path(info->dlpi_name);
  if (!dynlib_starts_with_ci(dynlib_basename(path), search->prefix)) {
    return 0;
  }
  void *handle = dynlib_loaded_module_with_symbol(path, search->symbol_name,
                                                  search->exclude);
  if (handle != nullptr) {
    search->matches.emplace_back(std::move(path), handle);
  }
  return 0;
}
#endif

inline std::vector<std::pair<std::string, void *>>
dynlib_loaded_modules_with_symbol_prefix(const char *prefix,
                                         const char *symbol_name,
                                         void *exclude = nullptr) {
#if defined(__linux__)
  DynlibLoadedSearch search{prefix, symbol_name, exclude, {}};
  dl_iterate_phdr(dynlib_collect_loaded_module, &search);
  return std::move(search.matches);
#else
  (void)prefix;
  (void)symbol_name;
  (void)exclude;
  return {};
#endif
}

inline void dynlib_release_loaded_probe_handle(void *handle, void *selected) {
#if defined(__linux__)
  if (handle != nullptr && handle != selected) {
    dlclose(handle);
  }
#else
  (void)handle;
  (void)selected;
#endif
}

#endif

inline void dynlib_append_existing_dir(std::vector<std::string> *dirs,
                                       const std::string &dir) {
  if (dynlib_is_dir(dir)) {
    dynlib_append_unique(dirs, dir);
  }
}

inline void dynlib_append_path_list(std::vector<std::string> *dirs,
                                    const std::string &path_list) {
  size_t start = 0;
  while (start <= path_list.size()) {
    size_t end = path_list.find(kDynlibPathListSeparator, start);
    std::string item = path_list.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    dynlib_append_existing_dir(dirs, item);
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
}

inline std::vector<std::string> dynlib_find_child_dirs(const std::string &dir,
                                                       const char *pattern) {
  std::vector<std::string> result;
  for (const std::string &path : dynlib_find_paths(dynlib_join(dir, pattern))) {
    if (dynlib_is_dir(path)) {
      dynlib_append_unique(&result, path);
    }
  }
  return result;
}

inline void dynlib_append_cuda_root_dirs(std::vector<std::string> *dirs,
                                         const std::string &root) {
  if (root.empty()) {
    return;
  }
  dynlib_append_existing_dir(dirs, root);
#if defined(_WIN32) && !defined(__CYGWIN__)
  dynlib_append_existing_dir(dirs, dynlib_join(root, "bin"));
  dynlib_append_existing_dir(dirs, dynlib_join(root, "bin\\x86_64"));
  dynlib_append_existing_dir(dirs, dynlib_join(root, "lib\\x64"));
  dynlib_append_existing_dir(dirs, dynlib_join(root, "nvvm\\bin"));
#else
  dynlib_append_existing_dir(dirs, dynlib_join(root, "lib64"));
  dynlib_append_existing_dir(dirs, dynlib_join(root, "lib"));
  for (const std::string &dir :
       dynlib_find_paths(dynlib_join(root, "targets/*/lib"))) {
    dynlib_append_existing_dir(dirs, dir);
  }
#endif
}

inline void dynlib_append_nvidia_package_dirs(std::vector<std::string> *dirs,
                                              const std::string &nvidia_root) {
  if (!dynlib_is_dir(nvidia_root)) {
    return;
  }
#if defined(_WIN32) && !defined(__CYGWIN__)
  for (const std::string &cu_root :
       dynlib_find_child_dirs(nvidia_root, "cu*")) {
    dynlib_append_cuda_root_dirs(dirs, cu_root);
  }
#else
  for (const char *pattern : {"*/lib", "*/lib64", "*/*/lib", "*/*/lib64"}) {
    for (const std::string &dir :
         dynlib_find_paths(dynlib_join(nvidia_root, pattern))) {
      dynlib_append_existing_dir(dirs, dir);
    }
  }
#endif
}

inline void
dynlib_append_site_packages_cuda_dirs(std::vector<std::string> *dirs,
                                      const std::string &site_packages) {
  dynlib_append_nvidia_package_dirs(dirs, dynlib_join(site_packages, "nvidia"));
}

inline void
dynlib_append_site_packages_from_path(std::vector<std::string> *dirs,
                                      const std::string &path) {
  for (const char *marker : {"/site-packages/", "\\site-packages\\",
                             "/dist-packages/", "\\dist-packages\\"}) {
    size_t pos = path.find(marker);
    if (pos != std::string::npos) {
      dynlib_append_site_packages_cuda_dirs(
          dirs, path.substr(0, pos + std::strlen(marker) - 1));
    }
  }
}

inline void dynlib_append_python_cuda_dirs(std::vector<std::string> *dirs,
                                           const std::string &python_root) {
  if (python_root.empty()) {
    return;
  }
#if defined(_WIN32) && !defined(__CYGWIN__)
  dynlib_append_site_packages_cuda_dirs(
      dirs, dynlib_join(dynlib_join(python_root, "Lib"), "site-packages"));
#else
  for (const char *lib_dir : {"lib", "lib64"}) {
    std::string base = dynlib_join(python_root, lib_dir);
    for (const std::string &site_packages :
         dynlib_find_paths(dynlib_join(base, "python*/site-packages"))) {
      dynlib_append_site_packages_cuda_dirs(dirs, site_packages);
    }
    for (const std::string &dist_packages :
         dynlib_find_paths(dynlib_join(base, "python*/dist-packages"))) {
      dynlib_append_site_packages_cuda_dirs(dirs, dist_packages);
    }
  }
  dynlib_append_site_packages_cuda_dirs(
      dirs, dynlib_join(dynlib_join(python_root, "lib"), "site-packages"));
#endif
}

#if defined(_WIN32) && !defined(__CYGWIN__)
inline void
dynlib_append_program_files_cuda_dirs(std::vector<std::string> *dirs) {
  for (const char *env : {"ProgramFiles", "ProgramW6432"}) {
    std::string program_files = dynlib_get_env(env);
    if (program_files.empty()) {
      continue;
    }
    std::string cuda_parent = dynlib_join(
        dynlib_join(program_files, "NVIDIA GPU Computing Toolkit"), "CUDA");
    for (const std::string &root : dynlib_find_child_dirs(cuda_parent, "v*")) {
      dynlib_append_cuda_root_dirs(dirs, root);
    }
  }
}
#endif

inline std::vector<std::string> dynlib_cuda_search_dirs() {
  std::vector<std::string> dirs;

  std::string module_path = dynlib_module_path_from_address(
      reinterpret_cast<const void *>(&dynlib_cuda_search_dirs));
  dynlib_append_existing_dir(&dirs, dynlib_dirname(module_path));
  dynlib_append_site_packages_from_path(&dirs, module_path);

  std::string exe_path = dynlib_exe_path();
  std::string exe_dir = dynlib_dirname(exe_path);
  dynlib_append_existing_dir(&dirs, exe_dir);
  dynlib_append_python_cuda_dirs(&dirs, dynlib_dirname(exe_dir));

  dynlib_append_cuda_root_dirs(&dirs, dynlib_get_env("CUDA_PATH"));
  dynlib_append_cuda_root_dirs(&dirs, dynlib_get_env("CUDA_HOME"));
  dynlib_append_python_cuda_dirs(&dirs, dynlib_get_env("VIRTUAL_ENV"));
  dynlib_append_python_cuda_dirs(&dirs, dynlib_get_env("CONDA_PREFIX"));

#if defined(_WIN32) && !defined(__CYGWIN__)
  dynlib_append_path_list(&dirs, dynlib_get_env("PATH"));
  dynlib_append_program_files_cuda_dirs(&dirs);
#else
  dynlib_append_path_list(&dirs, dynlib_get_env("LD_LIBRARY_PATH"));
  dynlib_append_cuda_root_dirs(&dirs, "/usr/local/cuda");
  for (const std::string &root : dynlib_find_paths("/usr/local/cuda-*")) {
    dynlib_append_cuda_root_dirs(&dirs, root);
  }
  dynlib_append_existing_dir(&dirs, "/usr/local/lib64");
  dynlib_append_existing_dir(&dirs, "/usr/local/lib");
  dynlib_append_existing_dir(&dirs, "/usr/lib64");
  dynlib_append_existing_dir(&dirs, "/usr/lib");
  dynlib_append_existing_dir(&dirs, "/usr/lib/x86_64-linux-gnu");
  dynlib_append_existing_dir(&dirs, "/usr/lib/aarch64-linux-gnu");
#endif

  return dirs;
}

inline std::vector<std::string>
dynlib_find_files_in_dirs(const std::vector<std::string> &dirs,
                          const char *filename_pattern) {
  std::vector<std::string> result;
  for (const std::string &dir : dirs) {
    for (const std::string &path :
         dynlib_find_paths(dynlib_join(dir, filename_pattern))) {
      if (dynlib_is_file(path)) {
        dynlib_append_unique(&result, path);
      }
    }
  }
  return result;
}

inline std::vector<unsigned long long>
dynlib_version_groups(const std::string &path, const char *prefix) {
  std::string name = dynlib_lower(dynlib_basename(path));
  std::string lower_prefix = dynlib_lower(prefix != nullptr ? prefix : "");
  size_t pos = name.find(lower_prefix);
  if (pos == std::string::npos) {
    return {};
  }
  pos += lower_prefix.size();

  std::vector<unsigned long long> groups;
  while (pos < name.size()) {
    while (pos < name.size() &&
           !std::isdigit(static_cast<unsigned char>(name[pos]))) {
      ++pos;
    }
    unsigned long long group = 0;
    size_t digits = 0;
    while (pos < name.size() &&
           std::isdigit(static_cast<unsigned char>(name[pos]))) {
      group = group * 10 + static_cast<unsigned long long>(name[pos] - '0');
      ++pos;
      ++digits;
    }
    if (digits > 0) {
      groups.push_back(group);
    }
  }
  return groups;
}

inline unsigned long long dynlib_library_version_key(const std::string &path,
                                                     const char *prefix) {
  std::vector<unsigned long long> groups = dynlib_version_groups(path, prefix);
  if (groups.empty()) {
    return 0;
  }

#if defined(_WIN32) && !defined(__CYGWIN__)
  unsigned long long encoded = groups[0];
  unsigned long long minor = 0;
  if (encoded >= 100) {
    minor = encoded % 10;
    encoded /= 10;
  }
  unsigned long long key = encoded * 10000 + minor * 100;
  for (size_t i = 1; i < groups.size(); ++i) {
    key += groups[i];
  }
  return key;
#else
  unsigned long long key = 0;
  int count = 0;
  for (unsigned long long group : groups) {
    if (count++ >= 4) {
      break;
    }
    key = key * 10000 + group;
  }
  while (count++ < 4) {
    key *= 10000;
  }
  return key;
#endif
}

inline void dynlib_sort_library_candidates(std::vector<std::string> *paths,
                                           const char *prefix) {
  std::sort(paths->begin(), paths->end(),
            [prefix](const std::string &lhs, const std::string &rhs) {
              unsigned long long lhs_key =
                  dynlib_library_version_key(lhs, prefix);
              unsigned long long rhs_key =
                  dynlib_library_version_key(rhs, prefix);
              if (lhs_key != rhs_key) {
                return lhs_key > rhs_key;
              }
#if defined(_WIN32) && !defined(__CYGWIN__)
              return dynlib_lower(lhs) > dynlib_lower(rhs);
#else
              return lhs > rhs;
#endif
            });
}

inline std::string dynlib_pattern_prefix(const char *filename_pattern) {
  std::string pattern = filename_pattern != nullptr ? filename_pattern : "";
  size_t wildcard = pattern.find_first_of("*?");
  return wildcard == std::string::npos ? pattern : pattern.substr(0, wildcard);
}

inline void *dynlib_find_loaded_by_basename_prefix(const char *library_prefix,
                                                   const char *symbol_name,
                                                   void *exclude = nullptr) {
  std::vector<std::pair<std::string, void *>> matches =
      dynlib_loaded_modules_with_symbol_prefix(library_prefix, symbol_name,
                                               exclude);
  std::sort(matches.begin(), matches.end(),
            [library_prefix](const auto &lhs, const auto &rhs) {
              unsigned long long lhs_key =
                  dynlib_library_version_key(lhs.first, library_prefix);
              unsigned long long rhs_key =
                  dynlib_library_version_key(rhs.first, library_prefix);
              if (lhs_key != rhs_key) {
                return lhs_key > rhs_key;
              }
#if defined(_WIN32) && !defined(__CYGWIN__)
              return dynlib_lower(lhs.first) > dynlib_lower(rhs.first);
#else
              return lhs.first > rhs.first;
#endif
            });
  if (matches.empty()) {
    return dynlib_find_loaded(symbol_name, exclude);
  }

  void *selected = matches.front().second;
  for (size_t i = 1; i < matches.size(); ++i) {
    dynlib_release_loaded_probe_handle(matches[i].second, selected);
  }
  return selected;
}

inline void *dynlib_open_first(const std::vector<std::string> &paths,
                               const char *symbol_name = nullptr) {
  for (const std::string &path : paths) {
    if (path.empty()) {
      continue;
    }
    void *handle = dynlib_open(path.c_str());
    if (handle == nullptr || symbol_name == nullptr) {
      if (handle != nullptr) {
        return handle;
      }
      continue;
    }
    if (dynlib_sym(handle, symbol_name) != nullptr) {
      return handle;
    }
    dynlib_close(handle);
  }
  return nullptr;
}

inline void *dynlib_open_matching(const char *filename_pattern,
                                  const char *symbol_name = nullptr) {
  std::vector<std::string> paths =
      dynlib_find_files_in_dirs(dynlib_cuda_search_dirs(), filename_pattern);
  std::string prefix = dynlib_pattern_prefix(filename_pattern);
  dynlib_sort_library_candidates(&paths, prefix.c_str());
  return dynlib_open_first(paths, symbol_name);
}

} // namespace tvm::tl::stubs
