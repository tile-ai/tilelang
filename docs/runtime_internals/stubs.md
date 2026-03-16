# CUDA and ROCm Stub Libraries

This document describes TileLang's stub mechanism for GPU driver/runtime
libraries (CUDA and ROCm/HIP).

## Purpose

CUDA:

1. **CUDA Driver (`cuda_stub`)**: Allows TileLang to be imported on systems
   without a GPU (e.g., CI/compilation nodes) by lazy-loading `libcuda.so` only
   when needed.
2. **CUDA Runtime & Compiler (`cudart_stub`, `nvrtc_stub`)**: Resolves SONAME
   versioning mismatches (e.g. `libcudart.so.11` vs `libcudart.so.12`),
   enabling a single build to work across different CUDA versions. This is
   achieved by reusing CUDA libraries already loaded by frameworks like PyTorch
   when possible.

ROCm:

1. **HIP Runtime/Module API (`hip_stub`)**: Allows TileLang to be imported on
   systems without ROCm installed by lazy-loading `libamdhip64.so` only when
   needed. The stub also prefers already-loaded symbols via `RTLD_DEFAULT` /
   `RTLD_NEXT` to interoperate with frameworks that have already loaded HIP.
2. **HIP Runtime Compiler (`hiprtc_stub`)**: Lazily loads `libhiprtc.so` and
   exposes the minimal HIPRTC API subset used by TileLang/TVM.

## Implementation

The stubs in `src/target/stubs/` implement a lazy-loading mechanism:

- **Lazy Loading**: Libraries are loaded via `dlopen` only upon the first API call.
- **Global Symbol Reuse**: For `cudart` and `nvrtc`, the stubs first check the global namespace (`RTLD_DEFAULT`) to use any already loaded symbols (e.g., from PyTorch).
- **ROCm Notes**: `hip_stub` checks `RTLD_DEFAULT` / `RTLD_NEXT` first and then
  falls back to `dlopen("libamdhip64.so")`. It additionally provides wrappers
  for `hsa_init` / `hsa_shut_down` so that ROCm-enabled wheels do not record a
  hard dependency on `libhsa-runtime64` at import time.
- **Versioning Support**: Handles ABI differences between CUDA versions (e.g., `cudaGraphInstantiate` changes in CUDA 12).

## Build Option

- `TILELANG_USE_CUDA_STUBS` (Default: `ON`) controls CUDA stubs. When enabled,
  TileLang links against these stubs instead of the system CUDA toolkit
  libraries.
- `TILELANG_USE_HIP_STUBS` (Default: `ON`) controls ROCm stubs. When enabled
  (and `USE_ROCM=ON`), TileLang/TVM link against `hip_stub` / `hiprtc_stub`
  instead of the system ROCm libraries.
