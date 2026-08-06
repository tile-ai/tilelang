# TileLang Backend Architecture

TileLang treats a backend as an end-to-end vertical slice, from its language
dialect to the code that loads and launches the generated kernel. The goal is
to make backend ownership explicit while keeping the common TileLang frontend
and compiler entry points backend-neutral.

Coding agents implementing a new backend must use the
[`tilelang-backend` skill](../../.agents/skills/tilelang-backend/SKILL.md).

## Design Model

A backend owns five related parts:

| Part | Responsibility |
| --- | --- |
| Language dialect | Extends the common TileLang language with backend-specific operations and intrinsics. |
| Context preparation | Normalizes the device and host targets, selects the target backend, and resolves an execution backend. |
| Pass pipeline | Lowers common and backend-specific IR through one explicit, backend-owned pass sequence. |
| Host/device codegen | Converts lowered host and device IR into source or runtime modules. |
| Execution backend | Owns the selected build, JIT, and runtime path: compiling artifacts, loading them, and launching kernels. |

The logical flow is:

```text
backend language dialect
        |
        v
    frontend IR
        |
        v
BackendContext preparation
        |
        v
backend PassPipeline
        |
        v
host/device split
        |
        +-----------> HostCodegen ----+
        |                             |
        +-----------> DeviceCodegen --+
                                      |
                                      v
                         selected ExecutionBackend
                           Build -> Load -> Launch
```

`BackendContext` is created once and accompanies the compilation through the
pipeline and codegen stages into the selected execution backend. A later stage
must not infer or resolve the backend again.

### Terminology

- A **target backend**, such as CUDA, ROCm, CPU, Metal, or WebGPU, owns the
  dialect, lowering, codegen, and target-specific toolchain primitives.
- An **execution backend**, such as `tvm_ffi`, `nvrtc`, `cython`, `torch`, or
  `cutedsl`, selects and implements a build/JIT/runtime path for a target
  backend.
- `BackendModule` is the compiler-facing registration manifest. It describes
  the components used by the current implementation, but it is not the whole
  backend implementation.
- `BackendContext` is the resolved, immutable state for one compilation.

Target backends and execution backends are deliberately separate. For example,
CUDA can execute through `tvm_ffi`, `nvrtc`, or `cython`; selecting `nvrtc` does
not select a different target architecture or pass pipeline.

## Source Layout

The Python implementation is split into common infrastructure and
backend-owned packages:

- `tilelang/backend/` contains the backend registry, component interfaces,
  context resolution, and small shared helpers.
- `tilelang/<backend>/` owns the dialect, target helpers, pipeline, codegen,
  toolchain hooks, operation implementations, intrinsics, and execution
  policies for a target backend.
- `tilelang/jit/` contains shared JIT infrastructure and the current execution
  adapters.

The native side mirrors target-backend ownership under `src/<backend>/`, where
C++ op lowering, codegen, runtime modules, toolchain stubs, and backend-local
CMake files live. `src/backend/` is reserved for shared native backend helpers.

## Backend Manifest

Each backend's `backend.py` publishes a `BackendModule`, the typed manifest used
by the compiler registry:

```python
BACKEND = register_backend(
    BackendModule(
        name="cuda",
        target_kinds=("cuda",),
        supports_target=is_plain_cuda_target,
        pipelines={...},
        device_codegens={...},
        host_codegens={...},
        host_codegen_hooks={...},
        execution_backends=(
            ExecutionBackendSpec("tvm_ffi"),
            ExecutionBackendSpec("nvrtc"),
        ),
        callbacks={...},
    )
)
```

The manifest currently declares:

- its name and owned TVM target kinds;
- an optional target predicate used to distinguish variants;
- one pass pipeline and one device-codegen entry for every owned target kind;
- optional host-codegen entries and pre-codegen hooks;
- the supported execution backends in `auto` preference order;
- FFI callbacks used by backend validation or target-toolchain integration.

`register_backend()` validates the complete declaration and publishes it once.
Backend packages are initialized during TileLang import, so the registry does
not maintain import paths, loading state, or synchronization.

Several manifests may share one TVM target kind when predicates make the
variants unambiguous. CUDA and CuTeDSL are separate manifests that both match
the `cuda` target kind. They intentionally reuse the CUDA pipeline but declare
different device codegen and execution policies. Pipeline reuse must be
explicit; it must not come from target-specific branching in the engine.

## Backend Context Preparation

The public compiler or JIT entry creates one immutable `BackendContext`:

```python
context = create_backend_context(target, target_host, execution_backend)
```

Context preparation performs three operations:

1. Normalize the device target and host target.
2. Select exactly one `BackendModule` using `target.kind.name` and, when
   needed, `supports_target`.
3. Resolve one available `ExecutionBackendSpec` from the user request or the
   backend's ordered `auto` preference.

The resulting context binds:

```text
BackendContext
  module             selected target-backend manifest
  target             normalized device target
  target_host        normalized host target
  execution_backend  selected build/JIT/runtime policy
```

Cache, lowering, codegen, and JIT code pass the same context instance.
Backend-specific target parsing and canonicalization belong in the backend
package; backend selection itself stays in the shared context factory.

## Language Dialects

`tilelang/language` defines the backend-neutral language surface. A backend may
freely compose that surface with extensions under
`tilelang/<backend>/language`:

```python
from tilelang import language as T       # common + CUDA compatibility facade
from tilelang.cuda import language as T  # common + CUDA extensions
from tilelang.rocm import language as T  # common + ROCm extensions
```

Dialect selection is static and follows the import path; it does not depend on
a process-global mutable dialect registry. Backend-specific language APIs emit
IR operations or annotations that the corresponding backend pipeline knows how
to lower. The language layer should describe semantics and construct IR, not
perform target code generation itself.

For example, CUDA owns WGMMA and TCGEN05 language helpers and intrinsic
emitters, while ROCm owns MFMA and WMMA helpers. Library code, tests, and
examples using backend-specific symbols should prefer explicit backend
language imports so static analysis and autocomplete can identify the dialect.

The language dialect is part of the logical backend, even though it is not a
field of the current `BackendModule` manifest.

## Pass Pipeline

Every target backend must explicitly own a complete lowering sequence after
the shared semantic checks:

```text
PreLowerSemanticCheck(mod)  # shared frontend boundary
mod = context.lower(mod)    # selected backend pipeline
```

The ordered pass list lives in `tilelang/<backend>/pipeline.py`. Backend-only
passes must be called there rather than dispatched from
`tilelang/engine/lower.py`. A backend pipeline may use small shared helpers from
`tilelang/backend/pass_pipeline`, but its ordering and target-specific choices
must remain visible in the backend package.

A target variant may deliberately reference another backend's pipeline when
their IR lowering is identical, as CuTeDSL currently does with CUDA. This is
explicit pipeline composition, not implicit engine behavior.

## Host and Device Codegen

After the pipeline, the compiler splits the module into host and device IR.
Both codegen paths are selected through `BackendContext`:

```text
device_mod = context.codegen_device(device_mod, compile_device=...)

host_mod = context.preprocess_host_codegen(host_mod)
host_mod = context.codegen_host(host_mod)
```

`DeviceCodegen` provides compiled and source-only entry points when the backend
supports both. `HostCodegen` is selected from the concrete host target, usually
`c` or `llvm`. `HostCodegenHook` lets a device backend prepare host IR without
adding target checks to the shared engine; Metal uses such a hook to mark
functions that require Metal runtime context.

The engine must not contain a `target.kind.name` dispatch table for backend
codegen. Backend variants, such as CUDA and CuTeDSL, declare their own codegen
entries in their manifests.

## Execution Backends: Build, JIT, and Runtime

An execution backend owns the complete path from generated host/device code to
a running kernel. Its implementation contains three related phases:

- **Build** invokes the required compiler and linker, applies architecture and
  toolchain options, and packages generated code into a loadable artifact such
  as a cubin, HSACO, shared library, or runtime module.
- **JIT** performs cache lookup, decides when to invoke Build, loads the
  artifact, creates wrappers, and returns a callable kernel adapter.
- **Runtime** owns the loaded module or kernel handle, binds arguments and
  streams, launches the kernel, and manages execution-time lifetime and errors.

Build is therefore an internal execution-backend stage rather than a separate
top-level backend component. The target backend supplies reusable codegen,
validation, compiler callbacks, and toolchain helpers; the execution backend
decides which of them to use and whether compilation is eager or deferred. For
example, `tvm_ffi` requests host codegen and eager device compilation, whereas
`nvrtc` consumes generated CUDA source and compiles it later in its adapter.

Conceptually, a target backend exposes a mapping such as:

```text
execution_backends:
  tvm_ffi -> TVM FFI build/load/launch path
  nvrtc   -> NVRTC compile + CUDA driver launch path
  cython  -> generated Cython wrapper path
  torch   -> framework-provided compile/load/launch path
```

`ExecutionBackendSpec` currently records the execution name, availability and
target predicates, and whether host codegen or eager device compilation is
required. The selected spec is stored in `BackendContext`.

Parts of CodeGen and Build are currently combined behind native
`target.build.*` functions and the historical `DeviceCodegen.build` name;
`build_without_compile` exposes the source-only path. These names describe the
current implementation, while the architectural ownership remains with the
selected execution backend.

The concrete adapters currently live under `tilelang/jit/adapter`, and adapter
dispatch still occurs in the shared JIT layer. Architecturally, however, the
spec selects a concrete Build/JIT/Runtime implementation. New execution paths
should move behavior behind that interface rather than add target-specific
decisions to lowering or codegen.

If TileLang adds a pure AOT/export workflow that produces artifacts without
loading or executing them, it may expose Build as a reusable interface. That
does not require Build to be a separate top-level component in the normal JIT
architecture.

## Registered Target Backends

| Python package | Target kind | Notes |
| --- | --- | --- |
| `tilelang/cuda/backend.py` | `cuda` | Plain CUDA codegen, compiler callbacks, and execution policies. |
| `tilelang/cuda/cutedsl_backend.py` | `cuda` | CuTeDSL variant explicitly reusing the CUDA pipeline. |
| `tilelang/rocm` | `hip` | ROCm/HIP pipeline, codegen, compiler callback, and MFMA/WMMA extensions. |
| `tilelang/cpu` | `c`, `llvm` | CPU pipeline, codegen, and scalar CPU tile-op implementations. |
| `tilelang/metal` | `metal` | Metal pipeline, codegen, host hook, and Metal language extensions. |
| `tilelang/webgpu/backend.py` | `webgpu` | WebGPU compiler component registration. |

## Common Backend Infrastructure

`tilelang/backend` should stay small and contain shared interfaces and
plumbing, not target-specific implementations:

```text
tilelang/backend/
  __init__.py
  module.py
  target.py
  device_codegen.py
  host_codegen.py
  execution_backend.py
  pass_pipeline/
    __init__.py
    pipeline.py
    pipeline_utils.py
```

- `module.py` defines `BackendModule`, `BackendContext`, registration,
  validation, and context resolution.
- `target.py` provides common target detection and normalization plumbing.
- `pass_pipeline/pipeline.py` defines `PassPipeline`.
- `device_codegen.py` and `host_codegen.py` define codegen component types and
  shared global-function helpers.
- `execution_backend.py` defines the current execution-backend selection and
  capability descriptor.
- `pass_pipeline/pipeline_utils.py` contains small shared helpers for pass
  configuration, visualization, vectorization gates, and shared-memory reuse.

## Backend Package Layout

A typical target-backend package has the following shape:

```text
tilelang/<backend>/
  __init__.py
  backend.py             manifest and backend toolchain callbacks
  target.py              target parsing and normalization helpers
  execution_backend.py   supported Build/JIT/Runtime paths and auto preference
  language/              backend language dialect
  pipeline.py            complete lowering sequence
  codegen.py             host/device codegen entries
  transform/             backend-only Python passes
  op/                    tile-op implementations and registration
  intrinsics/            backend intrinsic emitters and helpers
```

Not every backend needs every file. Component files define implementations;
`backend.py` is the single place that assembles and registers the manifest.
Import-time registration should remain deterministic and lightweight.

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

- `op/`: native tile-op lowering helpers;
- `transform/`: native backend-only passes;
- `codegen/`: target codegen, toolchain entry points, and runtime-module
  integration;
- `stubs/`: optional lazy-loading driver/runtime stubs;
- `CMakeLists.txt`: backend-local source selection and toolchain setup.

Shared native helpers with no target-runtime dependency belong in
`src/backend/common`.

## Ownership Rules

- Keep the common language surface and shared semantic checks backend-neutral.
- Put backend language extensions under `tilelang/<backend>/language` and lower
  them in that backend's pipeline.
- Resolve target and execution state once when creating `BackendContext`.
- Keep the complete backend-specific pass order in
  `tilelang/<backend>/pipeline.py`.
- Keep host/device codegen dispatch, host preparation hooks, compiler
  callbacks, and target-specific toolchain helpers under target-backend
  ownership.
- Treat execution-backend selection as the choice of a concrete
  Build/JIT/Runtime path; do not spread execution-mode checks through compiler
  passes.
- Keep backend registration explicit. Do not infer target ownership from a
  directory name: `tilelang/rocm`, for example, owns the TVM `hip` target kind.
- Register manifests during normal package initialization, without maintaining
  a second lazy-loading registry.
