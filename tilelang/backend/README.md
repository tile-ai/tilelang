# TileLang Backend Architecture

TileLang treats a target backend as a compiler vertical slice, from its
language dialect through target code generation. Generated code is handed to a
separate, reusable execution backend for Build, loading, and launch. The goal
is to make both ownership boundaries explicit while keeping the common
TileLang frontend and compiler entry points backend-neutral.

Coding agents implementing a new backend must use the
[`tilelang-backend` skill](../../.agents/skills/tilelang-backend/SKILL.md).

## Design Model

A target backend owns four implementation areas:

| Part | Responsibility |
| --- | --- |
| Language dialect | Extends the common TileLang language with backend-specific operations and intrinsics. |
| Context contribution | Defines target detection/normalization, target ownership, and compatible execution backends used by shared context resolution. |
| Pass pipeline | Lowers common and backend-specific IR through one explicit, backend-owned pass sequence. |
| Host/device codegen | Converts lowered host and device IR into source or runtime modules and provides target-specific toolchain hooks. |

The target backend also declares which shared execution backends can consume
its outputs. This is a compatibility declaration, not a fifth implementation
area. A new target backend normally reuses `tvm_ffi`, `nvrtc`, `cython`,
`torch`, or another existing execution implementation.

The logical flow is:

```text
target-backend language dialect
        |
        v
    frontend IR
        |
        v
shared BackendContext resolution
  - selects target backend
  - selects compatible ExecutionBackend
        |
        v
target-backend PassPipeline
        |
        v
host/device split
        |
        +-----------> HostCodegen ----+
        |                             |
        +-----------> DeviceCodegen --+
                                      |
                                      v
                       shared ExecutionBackend
                           Build -> Load -> Launch
```

`BackendContext` is created once and accompanies the compilation through the
pipeline and codegen stages. It also records the selected shared execution
backend, which consumes the generated outputs. A later stage must not infer or
resolve either backend again.

### Terminology

- A **target backend**, such as CUDA, ROCm, CPU, Metal, or WebGPU, owns the
  dialect, lowering, codegen, and target-specific toolchain primitives.
- An **execution backend**, such as `tvm_ffi`, `nvrtc`, `cython`, `torch`, or
  `cutedsl`, is a reusable Build/JIT/Runtime implementation that may serve one
  or more compatible target backends.
- A target backend **declares execution compatibility** and satisfies the
  selected implementation's source, artifact, and launch-metadata contract; it
  does not normally implement another JIT adapter or runtime.
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
  compatibility declarations for a target backend.
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
- the compatible shared execution backends in `auto` preference order;
- FFI callbacks used by backend validation or target-toolchain integration.

`register_backend()` validates the complete declaration and publishes it once.
Backend packages are initialized during TileLang import, so the registry does
not maintain import paths, loading state, or synchronization.

Several manifests may share one TVM target kind when predicates make the
variants unambiguous. CUDA and CuTeDSL are separate manifests that both match
the `cuda` target kind. They intentionally reuse the CUDA pipeline but declare
different device codegen and execution compatibility. Pipeline reuse must be
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
  execution_backend  selected shared Build/JIT/Runtime implementation
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

## Execution Backend Compatibility

A target backend normally does not implement JIT or runtime behavior. It
declares which shared execution backends can consume its generated outputs:

```text
execution_backends:
  tvm_ffi -> compatible with TVM FFI runtime modules
  nvrtc   -> compatible with CUDA source + driver launch
  cython  -> compatible with generated Cython wrappers
  torch   -> compatible with framework-provided compile/launch
```

`ExecutionBackendSpec` currently records the execution name, availability and
target predicates, and whether host codegen or eager device compilation is
required. A compatible target backend must provide the codegen mode and output
contract expected by the selected implementation:

- compiled or source-only device codegen as requested;
- host codegen when requested;
- source, artifact, global-symbol, argument, and launch metadata;
- target-specific validation, compiler callbacks, and toolchain helpers.

The shared execution backend owns cache orchestration, wrapper creation,
artifact loading, argument/stream binding, launch, and runtime lifetime. It may
invoke target-owned compiler callbacks during Build, but those callbacks do not
make JIT/Runtime a target-backend implementation area.

Only add a new execution backend when no existing implementation can consume
the target output or drive its runtime API. That is a separate integration
task. Concrete adapters currently live under `tilelang/jit/adapter`, while
adapter dispatch still occurs in the shared JIT layer; new execution behavior
should move behind that shared interface rather than into target passes or
codegen dispatch.

Parts of CodeGen and Build remain combined behind native `target.build.*`
functions and the historical `DeviceCodegen.build` name. Treat this as an
implementation detail at the boundary: target codegen supplies the operation,
and the selected execution backend decides whether and when it is invoked.

## Registered Target Backends

| Python package | Target kind | Notes |
| --- | --- | --- |
| `tilelang/cuda/backend.py` | `cuda` | Plain CUDA codegen, compiler callbacks, and execution compatibility. |
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
  execution_backend.py   compatible execution paths and auto preference
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
- Make target backends declare execution compatibility rather than reimplement
  JIT/Runtime. Add a new execution backend only for a new artifact or runtime
  contract.
- Do not spread execution-mode checks through compiler passes.
- Keep backend registration explicit. Do not infer target ownership from a
  directory name: `tilelang/rocm`, for example, owns the TVM `hip` target kind.
- Register manifests during normal package initialization, without maintaining
  a second lazy-loading registry.
