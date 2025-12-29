/**
 * \file cuda.cc
 * \brief Stub library for lazy loading libcuda.so at runtime.
 *
 * This allows tilelang to be imported on CPU-only machines without CUDA.
 */

#include "cuda.h"

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace tvm::tl::cuda {

namespace {

// Library names to try loading (in order of preference)
constexpr const char *kLibCudaPaths[] = {
    "libcuda.so.1", // Versioned library (most common)
    "libcuda.so",   // Unversioned library
};

/**
 * \brief Try to load libcuda.so from various paths.
 * \return The dlopen handle, or nullptr if loading failed.
 */
void *try_load_libcuda() {
  void *handle = nullptr;
  for (const char *path : kLibCudaPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

/**
 * \brief Get symbol from library handle, returning nullptr on failure.
 */
template <typename T> T get_symbol(void *handle, const char *name) {
  // Clear any existing error
  dlerror();
  void *sym = dlsym(handle, name);
  // Check for error (symbol could legitimately be nullptr in some cases)
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

/**
 * \brief Create and initialize the CUDADriverAPI singleton.
 *
 * This function loads libcuda.so and resolves all function symbols.
 * Required symbols that are missing will cause an exception.
 * Optional symbols that are missing will be set to nullptr.
 *
 * \param[out] available Set to true if libcuda.so was loaded successfully.
 * \return The initialized CUDADriverAPI instance.
 * \throws std::runtime_error if a required symbol is missing.
 */
CUDADriverAPI create_driver_api() {
  CUDADriverAPI api{};
  void *handle = CUDADriverAPI::get_handle();

  if (handle == nullptr) {
    return api;
  }

// Lookup required symbols - throw if missing
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);                    \
  if (api.name##_ == nullptr) {                                                \
    throw std::runtime_error(                                                  \
        std::string("Failed to load required CUDA driver symbol: ") + #name +  \
        ". Error: " + (dlerror() ? dlerror() : "unknown"));                    \
  }
  TILELANG_LIBCUDA_API_REQUIRED(LOOKUP_REQUIRED)
#undef LOOKUP_REQUIRED

// Lookup optional symbols - set to nullptr if missing (no throw)
#define LOOKUP_OPTIONAL(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);
  TILELANG_LIBCUDA_API_OPTIONAL(LOOKUP_OPTIONAL)
#undef LOOKUP_OPTIONAL

  return api;
}

} // namespace

void *CUDADriverAPI::get_handle() {
  // Static handle ensures library is loaded only once
  static void *handle = try_load_libcuda();
  return handle;
}

bool CUDADriverAPI::is_available() { return get_handle() != nullptr; }

CUDADriverAPI *CUDADriverAPI::get() {
  static CUDADriverAPI singleton = create_driver_api();

  if (!is_available()) {
    throw std::runtime_error(
        "CUDA driver library (libcuda.so) not found. "
        "Please ensure NVIDIA drivers are installed, or use CPU-only mode.");
  }

  return &singleton;
}

} // namespace tvm::tl::cuda
