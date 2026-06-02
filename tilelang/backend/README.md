# TileLang Backend Layout

This is a short draft of the current multi-backend layout. The main goal of
this refactor is to make backend ownership explicit while keeping the frontend
TileLang language surface backend-neutral.

## Overview

The Python backend layer is split into two parts:

- `tilelang/backend/`: common backend infrastructure, especially the `Backend`
  descriptor, backend registry, device-codegen helpers, pass-pipeline
  registration, and shared pipeline utilities.
- `tilelang/<backend>/`: backend-owned Python implementation files, such as
  backend descriptors, pass pipelines, tile-op implementation registration,
  callbacks, codegen hooks, and backend intrinsics.

The native side mirrors this split under `src/<backend>/`, where C++ op
lowering, codegen, runtime modules, stubs, and backend-local CMake files live.
`src/backend/` is reserved for shared native backend helpers.

## Lowering Entry

`tilelang/engine/lower.py` owns the high-level lowering entry. It runs
backend-independent semantic checks first, then resolves a `Backend` from the
TVM target kind:

```text
PreLowerSemanticCheck(mod)
backend = resolve_backend(target)
mod = backend.lower(mod, target)
codegen_mod = backend.codegen(device_mod, target, compile=enable_device_compile)
```

The resolver is implemented in `tilelang/backend/registry.py`. Backends register
a `Backend` descriptor at import time. The descriptor owns target matching,
pipeline lowering, device codegen hooks, optional host pre-codegen hooks,
callback registration, feature queries, JIT metadata, and CMake metadata.

The existing `PassPipeline` API remains available and is wrapped by
`Backend.pipeline` during migration.

## Target Registration

| Python package | Target kind | Notes |
| --- | --- | --- |
| `tilelang/cuda/backend.py` | `cuda` | CUDA-specific pass sequence, CUDA tile ops, MMA/WGMMA/TCGEN05 intrinsics, CUDA transform wrappers, CUDA compile callbacks. |
| `tilelang/rocm/backend.py` | `hip` | ROCm/HIP pass sequence, MFMA/WMMA tile-op implementations, HIP compile callback. |
| `tilelang/cpu/backend.py` | `c`, `llvm` | CPU pass sequence and scalar CPU tile-op implementations. |
| `tilelang/metal/backend.py` | `metal` | Metal pass sequence, Metal GEMM registration, Metal host pre-codegen hook. |
| `tilelang/backend/common.py` | `webgpu` | Temporary/common registration for targets that do not yet own a dedicated Python backend package. |

The backend package name does not have to match `target.kind.name`. ROCm is the
main example: the package is `tilelang/rocm`, but it registers target kind
`hip`.

## Backend Descriptor

`Backend` is a Python-side descriptor:

```python
Backend(
    name="cuda",
    target_kinds=("cuda",),
    pipeline=cuda_pipeline,
    device_codegen=cuda_device_codegen,
    device_codegen_without_compile=cuda_device_codegen_without_compile,
    register_callbacks=register_cuda_callbacks,
    features={"warp_size": target_get_warp_size},
    execution_backends={
        "tvm_ffi": ExecutionBackendSpec(
            "tvm_ffi",
            enable_host_codegen=True,
            enable_device_compile=True,
        ),
        "cython": ExecutionBackendSpec("cython"),
        "nvrtc": ExecutionBackendSpec("nvrtc", is_available=is_nvrtc_available),
    },
    default_execution_backend="tvm_ffi",
    cmake_name="CUDA",
)
```

The registry resolves exactly one backend for a concrete TVM target. If multiple
backends match the same target kind, `priority` must break the tie; otherwise
resolution fails with an explicit ambiguity error.

The backend descriptor also owns target-dependent JIT execution policy. For
example, CUDA declares `nvrtc`, ROCm does not; CPU can choose `cython` by
default; and a backend can choose whether an execution mode requests host
codegen and device compilation.

Lazy import is supported through `register_lazy_backend(target_kind,
import_path)`, so optional backends can delay importing toolchain-specific Python
modules until a matching target is requested.

## TVM/tvm-ffi Boundary

The `Backend` registry is intentionally Python-side because it stores Python
callables, lazy import policy, feature query functions, and diagnostics.

TVM/tvm-ffi remains the right boundary for cross-language registration:

- Python compile callbacks use `tvm_ffi.register_global_func`, for example
  `tilelang_callback_cuda_compile` and `tilelang_callback_hip_compile`.
- Native transforms and codegen entry points stay registered from C++ through
  TVM FFI global functions, for example `target.build.tilelang_cuda`.
- Python `_ffi_api.py` modules should continue using `tvm_ffi.init_ffi_api`.

The first implementation should not make `Backend` a TVM FFI `ObjectRef`;
there is no native enumeration requirement yet, and doing so would make the
Python orchestration path more complex.

## `tilelang/backend`

`tilelang/backend` should stay small. It contains shared backend plumbing, not
backend-specific implementation details.

```text
tilelang/backend/
  __init__.py
  backend.py
  common.py
  codegen.py
  registry.py
  pass_pipeline/
    __init__.py
    pipeline.py
    pipeline_utils.py
```

- `backend.py` defines the `Backend` descriptor and hook types.
- `registry.py` defines `register_backend`, `register_lazy_backend`,
  `resolve_backend`, `get_backend`, and `list_backends`.
- `codegen.py` contains shared device-codegen cleanup helpers.
- `pass_pipeline/pipeline.py` defines `PassPipeline`, `register_pipeline`, and
  `resolve_pipeline`.
- `pass_pipeline/pipeline_utils.py` contains small shared helpers for pass
  configuration, layout visualization, vectorization gates, and shared-memory
  reuse flags.
- `common.py` registers shared fallback pipelines for target kinds that do not
  yet have a fully dedicated package.

Backend-specific pass lists should not live here. They should live in the
backend package that owns the target.

## Backend Packages

Each backend package owns the Python pieces needed to lower and register code
for that backend.

```text
tilelang/cuda/
  pipeline.py
  transform/
  op/
  intrinsics/

tilelang/rocm/
  pipeline.py
  op/
  intrinsics/

tilelang/cpu/
  pipeline.py
  op/

tilelang/metal/
  pipeline.py
  transform/
  op/
  intrinsics/
```

The `pipeline.py` file should expose one complete backend pass sequence after
semantic checking. It may use shared helpers from `tilelang/backend`, but the
ordered pass list should be visible in the backend-owned file. CUDA-only,
ROCm-only, and Metal-only passes should be called from the corresponding
backend pipeline rather than from engine-level code.

The `op/` and `intrinsics/` folders contain Python implementation and helper
code used by tile-op lowering. For example, CUDA owns MMA/WGMMA/TCGEN05
intrinsic emitters, while ROCm owns MFMA/WMMA emitters. Backend-local
transform passes, such as Metal's simdgroup lowering and host-context marking,
should live under that backend's `transform/` package.

## Native Backend Layout

Backend-specific native implementation lives directly under `src/<backend>`:

```text
src/
  cpu/
  cuda/
  metal/
  rocm/
  webgpu/
src/backend/
  common/
```

Typical backend-local subdirectories are:

- `op/`: native tile-op lowering helpers.
- `codegen/`: backend codegen and runtime module integration.
- `stubs/`: optional lazy-loading driver/runtime stubs for GPU backends.
- `CMakeLists.txt`: backend-local source selection and toolchain setup.

Shared native helpers that have no target runtime dependency belong in
`src/backend/common`.

## Guidelines

- Keep `tilelang/language` and `tilelang/tileop` backend-neutral.
- Keep backend-specific pass ordering in the backend package.
- Register backend implementations at import time, but keep import-time work
  light.
- Prefer explicit target-kind registration over implicit folder-name matching,
  because some names differ, such as `tilelang/rocm` registering target kind
  `hip`.
- When adding a backend-specific pass, put the call site in that backend's
  `pipeline.py` and keep only small shared predicates in `pipeline_utils.py`.
