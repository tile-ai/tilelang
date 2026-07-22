# TileLang Backend Layout

This is a short draft of the current multi-backend layout. The main goal of
this refactor is to make backend ownership explicit while keeping the frontend
TileLang language surface backend-neutral.

## Overview

The Python backend layer is split into two parts:

- `tilelang/backend/`: common backend infrastructure, especially pass-pipeline
  registration, host/device-codegen registration, and shared pipeline
  utilities.
- `tilelang/<backend>/`: backend-owned Python implementation files, such as
  pass pipelines, host/device-codegen entry registration, tile-op
  implementation registration, and backend intrinsics.

The native side mirrors this split under `src/<backend>/`, where C++ op
lowering, codegen, runtime modules, stubs, and backend-local CMake files live.
`src/backend/` is reserved for shared native backend helpers.

## Lowering Entry

`tilelang/engine/lower.py` owns the high-level lowering entry. It runs
backend-independent semantic checks first, then resolves a pass pipeline from
the TVM target kind:

```text
PreLowerSemanticCheck(mod)
pipeline = resolve_pipeline(target)
mod = pipeline.lower(mod, target)
```

The resolver is implemented in `tilelang/backend/pass_pipeline/pipeline.py`.
Backends register a `PassPipeline(name, lower)` at import time. The pipeline
name should match `target.kind.name`.

Device codegen follows the same ownership model after host/device splitting:

```text
codegen = resolve_device_codegen(target)
device_mod = codegen.lower(device_mod, target, compile_device=...)
```

The resolver is implemented in `tilelang/backend/device_codegen.py`. Backend
packages register one or more `DeviceCodegen` entries for their target kind.
CUDA, for example, owns both the plain CUDA entry and the CuTeDSL target
variant, while CPU owns the `c` and `llvm` entries. The engine-level lowering
code should not keep backend-specific `target.kind.name` dispatch for device
codegen.

Host codegen is resolved from the host target in the same style:

```text
host_mod = apply_host_codegen_hooks(host_mod, target_host, target)
codegen = resolve_host_codegen(target_host)
host_mod = codegen.lower(host_mod, target_host)
```

The resolver is implemented in `tilelang/backend/host_codegen.py`. Host build
entries are registered by the package that owns the host target kind, such as
`tilelang/cpu` for `c` and `llvm`. Device backends may also register host
codegen hooks for target-specific host preparation; Metal uses this to mark
host functions that need Metal runtime context.

## Target Registration

| Python package | Target kind | Notes |
| --- | --- | --- |
| `tilelang/cuda` | `cuda` | CUDA-specific pass sequence, CUDA tile ops, MMA/WGMMA/TCGEN05 intrinsics, CUDA transform wrappers. |
| `tilelang/rocm` | `hip` | ROCm/HIP pass sequence and MFMA/WMMA tile-op implementations. |
| `tilelang/cpu` | `c`, `llvm` | CPU pass sequence and scalar CPU tile-op implementations. |
| `tilelang/metal` | `metal` | Metal pass sequence and Metal GEMM registration. |
| `tilelang/backend/common.py` | `webgpu` | Temporary/common registration for targets that do not yet own a dedicated Python backend package. |

## `tilelang/backend`

`tilelang/backend` should stay small. It contains shared backend plumbing, not
backend-specific implementation details.

```text
tilelang/backend/
  __init__.py
  common.py
  device_codegen.py
  host_codegen.py
  pass_pipeline/
    __init__.py
    pipeline.py
    pipeline_utils.py
```

- `pass_pipeline/pipeline.py` defines `PassPipeline`, `register_pipeline`, and
  `resolve_pipeline`.
- `device_codegen.py` defines `DeviceCodegen`, `register_device_codegen`, and
  `resolve_device_codegen`.
- `host_codegen.py` defines `HostCodegen`, host codegen hooks, and
  `resolve_host_codegen`.
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
  codegen.py
  pipeline.py
  transform/
  op/
  intrinsics/

tilelang/rocm/
  codegen.py
  pipeline.py
  op/
  intrinsics/

tilelang/cpu/
  codegen.py
  pipeline.py
  op/

tilelang/metal/
  codegen.py
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

The `codegen.py` file should register the backend-owned host/device codegen
entry points, usually by mapping the target kind to native `target.build.*`
global functions. Target variants should be represented by backend-owned
predicates, not by engine-level branching.

The `op/` and `intrinsics/` folders contain Python implementation and helper
code used by tile-op lowering. For example, CUDA owns MMA/WGMMA/TCGEN05
intrinsic emitters, while ROCm owns MFMA/WMMA emitters. Backend-local
transform passes, such as Metal's simdgroup lowering and host-context marking,
should live under that backend's `transform/` package.

## Language Dialects

`tilelang/language` is the core language namespace. It should stay
backend-neutral over time, with legacy backend-specific exports treated as
compatibility shims.

Backend-specific language surfaces live under the backend package:

```python
from tilelang import language as T       # default/core language
from tilelang.cuda import language as T  # core + CUDA dialect
from tilelang.rocm import language as T  # core + ROCm dialect
```

Each backend language module re-exports the core language and adds only the
symbols owned by that backend. For example, CUDA exposes `T.tcgen05_mma`,
WGMMA/TCGEN05 helpers, and CUDA intrinsic emitters; ROCm exposes MFMA/WMMA
helpers.

The default used by `from tilelang import language as T` can be overridden for
interactive or legacy workflows:

```bash
TILELANG_DEFAULT_DIALECT=cuda
```

Library code, tests, and examples that rely on backend-specific symbols should
prefer explicit imports such as `from tilelang.cuda import language as T` so
static analysis and autocomplete can resolve the intended dialect.

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
- Keep backend-specific host-codegen dispatch and host preparation hooks in the
  backend package.
- Keep backend-specific device-codegen dispatch in the backend package.
- Register backend implementations at import time, but keep import-time work
  light.
- Prefer explicit target-kind registration over implicit folder-name matching,
  because some names differ, such as `tilelang/rocm` registering target kind
  `hip`.
- When adding a backend-specific pass, put the call site in that backend's
  `pipeline.py` and keep only small shared predicates in `pipeline_utils.py`.
