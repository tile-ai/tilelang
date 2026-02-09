# CUDA/CUDART/NVRTC Stubs

This document describes the stub mechanism used in TileLang for CUDA, CUDA Runtime (CUDART), and NVRTC libraries.

## Purpose

The primary purpose of these stubs is to decouple the TileLang build from specific versions of CUDA libraries and to allow TileLang to be imported on systems without CUDA installed.

1.  **Portability across CUDA versions**:
    -   CUDA libraries like `libcudart.so` and `libnvrtc.so` encode their major version in their SONAME (e.g., `libcudart.so.11.0`, `libcudart.so.12`).
    -   Linking directly against a specific version would make the resulting binary incompatible with systems having a different CUDA major version.
    -   By using stubs, TileLang loads the available library at runtime, allowing a single build to work with multiple CUDA versions (e.g., CUDA 11 and 12).

2.  **CPU-only Import**:
    -   TileLang can be installed and imported on machines without any CUDA drivers or toolkit installed (e.g., for compilation only or CI environments).
    -   The actual libraries are only loaded when a CUDA-related function is called.

## Components

The stub implementation is located in `src/target/stubs/` and consists of three main components:

### 1. CUDA Driver API Stub (`cuda.cc` / `cuda.h`)
-   **File**: `src/target/stubs/cuda.cc`, `src/target/stubs/cuda.h`
-   **Library**: `libcuda.so.1` (or `libcuda.so`)
-   **Description**: Provides stubs for the CUDA Driver API (e.g., `cuModuleLoad`, `cuLaunchKernel`). This is used for low-level device management and kernel launching.
-   **Implementation**:
    -   Defines a set of function pointers for required driver API symbols.
    -   Implements global wrapper functions (e.g., `cuGetErrorString`) that delegate to the loaded library.
    -   Uses `dlopen` to load `libcuda.so.1` on the first call to any driver API function.

### 2. CUDA Runtime API Stub (`cudart.cc`)
-   **File**: `src/target/stubs/cudart.cc`
-   **Library**: `libcudart.so`
-   **Description**: Provides stubs for the CUDA Runtime API (e.g., `cudaMalloc`, `cudaMemcpy`).
-   **Implementation**:
    -   First checks if symbols are already available in the global namespace (e.g., if PyTorch has already loaded `libcudart.so`).
    -   If not found globally, attempts to load `libcudart.so` via `dlopen`.
    -   Resolves symbols using `dlsym`.
    -   Includes logic to handle API changes between CUDA 11 and 12 (e.g., `cudaGraphInstantiate`).

### 3. NVRTC Stub (`nvrtc.cc`)
-   **File**: `src/target/stubs/nvrtc.cc`
-   **Library**: `libnvrtc.so`
-   **Description**: Provides stubs for the NVIDIA Runtime Compilation (NVRTC) library, used for JIT compilation of CUDA kernels.
-   **Implementation**:
    -   Similar to `cudart.cc`, it first checks for global symbols.
    -   Lazy loads `libnvrtc.so` if needed.

## Implementation Details

### Lazy Loading
All stubs use a lazy loading mechanism. The shared library is not loaded until the first call to a stubbed function.

1.  **Singleton Pattern**: A singleton class (e.g., `CUDADriverAPI`, `CUDARuntimeAPI`, `NVRTCAPI`) holds the library handle and function pointers.
2.  **Initialization**: The `Get*API()` function initializes the singleton on the first call. It calls `dlopen` to load the library and `dlsym` to resolve symbols.
3.  **Dispatch**: Global API functions (extern "C") retrieve the singleton and call the resolved function pointer.

```cpp
// Example from cudart.cc
cudaError_t cudaMalloc(void **devPtr, size_t size) {
  auto *api = GetCUDARuntimeAPI(); // Triggers load if not already loaded
  if (api->cudaMalloc_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaMalloc_(devPtr, size);
}
```

### Global Symbol Lookup
For `cudart` and `nvrtc`, the stubs first attempt to find symbols in the global namespace (`RTLD_DEFAULT`). This is crucial for interoperability with other libraries like PyTorch, which might have already loaded a specific version of CUDA libraries.

### Error Handling
If the library cannot be loaded or a symbol is missing:
-   The stub returns a specific error code (e.g., `cudaErrorUnknown`, `NVRTC_ERROR_INTERNAL_ERROR`).
-   It may print an error message to stderr explaining that the library was not found.

## Build System

The use of stubs is controlled by the CMake option `TILELANG_USE_CUDA_STUBS`.

-   **ON (Default)**: TileLang links against the stub libraries. This is recommended for distribution wheels.
-   **OFF**: TileLang links directly against the CUDA libraries found in the build environment.

When `TILELANG_USE_CUDA_STUBS` is ON:
-   The build defines `TILELANG_USE_CUDA_STUBS`.
-   Stub sources (`cuda.cc`, `cudart.cc`, `nvrtc.cc`) are compiled and linked.
-   Direct linking to CUDA toolkits is avoided.
