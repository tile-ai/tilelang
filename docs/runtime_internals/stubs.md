# CUDA/CUDART/NVRTC Stubs

This document describes the stub mechanism in TileLang for CUDA-related libraries.

## Purpose

1. **CUDA Driver (`cuda`)**: Allows TileLang to be imported on systems without a GPU (e.g., CI/compilation nodes) by lazy-loading `libcuda.so` only when needed.
2. **Runtime & Compiler (`cudart`, `nvrtc`)**: Resolves SONAME versioning mismatches, enabling a single build to work across different CUDA versions. This is achieved by reusing the existing CUDA runtime loaded by frameworks like PyTorch.

## Implementation

The stubs in `src/target/stubs/` implement a lazy-loading mechanism:

- **Lazy Loading**: Libraries are loaded via `dlopen` only upon the first API call.
- **Global Symbol Reuse**: For `cudart` and `nvrtc`, the stubs first check the global namespace (`RTLD_DEFAULT`) to use any already loaded symbols (e.g., from PyTorch).
- **Versioning Support**: Handles ABI differences between CUDA versions (e.g., `cudaGraphInstantiate` changes in CUDA 12).

## Build Option

`TILELANG_USE_CUDA_STUBS` (Default: `ON`) controls this behavior. When enabled, TileLang links against these stubs instead of the system CUDA toolkit.
