# CUDA, ROCm, and Ascend Stub Libraries

This document describes TileLang's stub mechanism for GPU/NPU driver and
runtime libraries (CUDA, ROCm/HIP, and Ascend/CANN).

## Purpose

CUDA:

1. **CUDA Driver (`cuda_stub`, file `libstub_cuda.so`)**: Allows TileLang to be
   imported on systems without a GPU (e.g., CI/compilation nodes) by
   lazy-loading `libcuda.so` only when needed.
2. **CUDA Runtime & Compiler (`cudart_stub`, file `libstub_cudart.so`;
   `nvrtc_stub`, file `libstub_nvrtc.so`)**: Resolves SONAME
   versioning mismatches (e.g. `libcudart.so.11` vs `libcudart.so.12`),
   enabling a single build to work across different CUDA versions. This is
   achieved by reusing CUDA libraries already loaded by frameworks like PyTorch
   when possible.

ROCm:

1. **HIP Runtime/Module API (`hip_stub`, file `libstub_hip.so`)**: Allows
   TileLang to be imported on systems without ROCm installed by lazy-loading
   `libamdhip64.so` only when needed. The stub also prefers already-loaded
   symbols via `RTLD_DEFAULT` / `RTLD_NEXT` to interoperate with frameworks that
   have already loaded HIP.
2. **HIP Runtime Compiler (`hiprtc_stub`, file `libstub_hiprtc.so`)**: Lazily
   loads `libhiprtc.so` and exposes the minimal HIPRTC API subset used by
   TileLang/TVM.

Ascend:

1. **CANN Runtime (`ascendcl_stub`, file `libstub_ascendcl.so`)**: Allows
   TileLang to be imported on systems without CANN installed by lazy-loading
   `libascendcl.so` only when needed. The stub prefers the copy already loaded
   by torch_npu (via `RTLD_DEFAULT` / `RTLD_NEXT`) before searching the
   filesystem. There is no RTC-style stub for Ascend: device kernels are
   compiled by invoking the `bisheng` compiler as a subprocess, not through a
   library API.

## Implementation

The CUDA stubs in `src/cuda/stubs/`, ROCm stubs in `src/rocm/stubs/`, and
Ascend stubs in `src/ascend/stubs/` implement a lazy-loading mechanism:

- **Lazy Loading**: Libraries are loaded via `dlopen` only upon the first API call.
- **Global Symbol Reuse**: For `cudart` and `nvrtc`, the stubs first check the global namespace (`RTLD_DEFAULT`) to use any already loaded symbols (e.g., from PyTorch).
- **ROCm Notes**: `hip_stub` checks `RTLD_DEFAULT` / `RTLD_NEXT` first and then
  falls back to `dlopen("libamdhip64.so")`. Device count queries return a HIP
  error code when the runtime or required symbols are missing, allowing
  `tvm.rocm(0).exist` to return `False` without throwing. The ROCm device
  query uses HIP initialization and has no direct HSA dependency.
- **Ascend Notes**: `ascendcl_stub` checks `RTLD_DEFAULT` / `RTLD_NEXT` first
  (reusing the `libascendcl` copy loaded by torch_npu), then
  `dlopen("libascendcl.so")` (which honors the `LD_LIBRARY_PATH` set by CANN's
  `set_env.sh`), and finally well-known install roots: `$ASCEND_HOME_PATH`,
  `$ASCEND_TOOLKIT_HOME`, and `/usr/local/Ascend/ascend-toolkit/latest`, each
  with the `lib64/` and `runtime/lib64/` suffixes. The stub exposes only the
  small entrypoint set used by the Ascend runtime module; `aclGetRecentErrMsg`
  degrades gracefully (returns `nullptr` instead of throwing) so error
  reporting never fails. No CANN headers are needed to build the stub — the
  ABI is expressed with opaque pointer and fixed-width integer types.
- **Versioning Support**: Handles ABI differences between CUDA versions (e.g., `cudaGraphInstantiate` changes in CUDA 12).

## Build Option

- `TILELANG_USE_CUDA_STUBS` (Default: `ON`) controls CUDA stubs. When enabled,
  TileLang links against these stubs instead of the system CUDA toolkit
  libraries.
- `TILELANG_USE_HIP_STUBS` (Default: `ON`) controls ROCm stubs. When enabled
  (and `USE_ROCM=ON`), TileLang/TVM link against `hip_stub` / `hiprtc_stub`
  instead of the system ROCm libraries.
- `TILELANG_USE_ASCEND_STUBS` (Default: `ON` on non-Windows) controls the
  Ascend stub. When enabled (and `USE_ASCEND=ON`), TileLang links against
  `ascendcl_stub` instead of the system CANN library. When disabled, TileLang
  links directly against the real `libascendcl` found under
  `$ASCEND_HOME_PATH` / `$ASCEND_TOOLKIT_HOME` /
  `/usr/local/Ascend/ascend-toolkit/latest`.
