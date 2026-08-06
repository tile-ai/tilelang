# TileLang Backend Integration Checklist

Use this reference with the architecture in `tilelang/backend/README.md`.

## Contents

1. [Scope Matrix](#scope-matrix)
2. [Preflight Decisions](#preflight-decisions)
3. [Language Dialect](#1-language-dialect)
4. [Backend Context Preparation](#2-backend-context-preparation)
5. [Pass Pipeline](#3-pass-pipeline)
6. [Host and Device Codegen](#4-host-and-device-codegen)
7. [Execution Backend Compatibility](#execution-backend-compatibility)
8. [Native Compilation and Packaging](#native-compilation-and-packaging)
9. [Tests and Documentation](#tests-and-documentation)
10. [Definition of Done](#definition-of-done)

## Scope Matrix

| Change | Required integration |
| --- | --- |
| New target backend | Four target-backend areas, execution compatibility declaration, imports, native compilation, packaging, tests, and docs. |
| Variant sharing a target kind | Separate manifest and predicate; explicit component reuse; variant-resolution tests. |
| New execution mode | Build/JIT/Runtime implementation, compatible manifests, shared dispatch bridge, cache/export/autotuning coverage. |

A package name may differ from its target kind. For example, `rocm` owns
`hip`. Decide the backend name, package name, target kind, and execution names
before creating files.

## Preflight Decisions

Record these choices in the implementation plan:

- TVM target kind and whether TVM already registers it;
- required target attributes and accepted aliases;
- auto-detection policy and architecture defaults;
- variant-selection predicate if the target kind is shared;
- common and backend-specific language surface;
- initial supported tile operations and explicit exclusions;
- SIMT, serial, or custom launch/thread model;
- source or runtime-module format;
- host targets and host preparation requirements;
- execution paths and `auto` preference;
- compatible execution implementations and eager/deferred compilation policy;
- hardware-free source-generation coverage;
- optional SDK/runtime and portable-wheel strategy.

Do not begin by copying the CUDA backend wholesale. Select the closest backend
for each layer; one backend may be the best pipeline reference while another is
the best execution reference.

## 1. Language Dialect

Create `tilelang/<backend>/language/__init__.py`, including for a common-only
initial dialect:

```python
from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

__tilelang_dialect__ = "<backend>"
__all__ = tuple(_COMMON_ALL)

del _COMMON_ALL
```

For extensions:

- place APIs under `tilelang/<backend>/language/`;
- export a deterministic `__all__` and set `__tilelang_dialect__`;
- construct IR operations, calls, or annotations instead of emitting source;
- add pipeline lowering or legalization for every emitted form;
- place Python intrinsic emitters in `tilelang/<backend>/intrinsics/` when used
  by tile-op selection;
- register native TIR builtins in `src/<backend>/op/` when required;
- use `from tilelang.<backend> import language as T` in tests/examples.

Do not add backend-only symbols to `tilelang.language.common` or the default
CUDA-compatible `tilelang.language` facade. Test every extension from dialect
construction through lowering.

## 2. Backend Context Preparation

### Target handling

Add `tilelang/<backend>/target.py` when detection, normalization, aliases,
architecture defaults, or predicates are needed.

- Use `register_target_detector()` only for real auto-detection.
- Use `register_target_normalizer()` for aliases and canonical attributes.
- Return `None` for inputs belonging to another backend.
- Preserve explicit user attributes unless documented canonicalization requires
  a change.
- Delay optional imports and hardware probes until the detector/normalizer is
  called.
- Make missing optional dependencies actionable.
- Support explicit target resolution even if auto-detection is unavailable.

Test strings, config dictionaries, `Target` objects, invalid attributes,
aliases, canonical attributes, and auto-detection when declared.

### Manifest

Create `tilelang/<backend>/backend.py`:

```python
BACKEND = register_backend(
    BackendModule(
        name="<backend>",
        target_kinds=("<target-kind>",),
        supports_target=...,  # required for a shared target kind
        pipelines={"<target-kind>": BACKEND_PIPELINE},
        device_codegens={"<target-kind>": BACKEND_DEVICE_CODEGEN},
        host_codegens=STANDARD_HOST_CODEGENS,
        host_codegen_hooks={...},
        execution_backends=EXECUTION_BACKENDS,
        callbacks={...},
    )
)
```

Honor `BackendModule` validation:

- `pipelines` and `device_codegens` contain exactly one entry per owned kind;
- each pipeline name equals its TVM target kind;
- at least one execution backend is declared;
- execution names are unique and declaration order defines `auto`;
- host codegens exist when an execution mode enables host codegen;
- every manifest sharing a target kind has a mutually exclusive predicate;
- FFI callback names are process-global and unique.

Use callbacks for backend-owned validation or compiler/toolchain integration,
not arbitrary engine dispatch.

### Package initialization

Create `tilelang/<backend>/__init__.py` with a cycle-safe order. Usually import:

1. target registration;
2. backend transforms;
3. language dialect;
4. intrinsics;
5. tile-op implementations;
6. backend manifest.

Import the package from `tilelang/__init__.py` so built-in backends register
during normal initialization. Test a fresh-process import without target
hardware or SDK. Optional external plugins may use a different loading model,
but must document and test it explicitly.

## 3. Pass Pipeline

Create `tilelang/<backend>/pipeline.py` with a complete ordered sequence after
the shared `PreLowerSemanticCheck`. Wrap it in a `PassPipeline` named for the
TVM target kind.

Deliberately cover these boundaries where applicable:

1. Bind the normalized target.
2. Materialize the kernel launch model.
3. Normalize and validate frontend IR.
4. Run pre-layout target transformations whose annotations affect layout.
5. Infer layouts and lower tile operations.
6. Legalize memory, vectorization, types, and target intrinsics.
7. Plan allocations and lower opaque blocks and buffers.
8. Verify memory and lower supported reductions/synchronization.
9. Split host and device functions.
10. Apply post-split storage, synchronization, packed API, and launch lowering.

Use shared helpers from `tilelang/backend/pass_pipeline` only for genuinely
common behavior. Keep the complete ordered list visible in the backend package.
Put Python backend-only passes in `tilelang/<backend>/transform/` and native
passes in `src/<backend>/transform/`.

Reference guidance:

- CPU: serial/non-SIMT launch handling;
- ROCm/Metal/WebGPU: comparatively direct SIMT pipelines;
- CUDA: CUDA-specific async memory, specialization, and architecture passes;
- CuTeDSL: explicit pipeline reuse by a target variant.

Reuse a pipeline only when lowering is identical, and add an identity/reuse
test. Never select a target-specific pipeline in `tilelang/engine/lower.py`.

## 4. Host and Device Codegen

### Device codegen

Create `tilelang/<backend>/codegen.py` and a `DeviceCodegen`. Native backends
usually register:

```text
target.build.tilelang_<backend>
target.build.tilelang_<backend>_without_compile
```

The compiled path emits target code and returns a loadable runtime module. The
source-only path emits the same source without invoking the target compiler.
Implement both when any execution path defers Build. If a mode is unsupported,
leave the callback absent and ensure no execution backend requests it.

Preserve downstream metadata:

- inspectable source;
- global symbols;
- argument types;
- launch parameter tags;
- target format;
- runtime-module metadata needed to load and invoke kernels.

Put native source generation and runtime-module integration under
`src/<backend>/codegen/`. Keep Python wrappers thin and map them to
backend-owned global functions.

### Host codegen

Reuse `STANDARD_HOST_CODEGENS` for `c`/`llvm` when compatible. Add a custom
`HostCodegen` only for a genuinely different host representation. Use
`HostCodegenHook` for device-backend preparation immediately before host
codegen instead of adding engine branches.

### Operations, intrinsics, and templates

Inventory every operation used by the first supported kernels:

- Python tile-op selectors: `tilelang/<backend>/op/`;
- native op lowerers: `src/<backend>/op/` with target predicates;
- Python intrinsic helpers: `tilelang/<backend>/intrinsics/`;
- native builtins/lowering: `src/<backend>/op/` or backend transforms;
- generated source templates: `src/tl_templates/<backend>/`.

Do not silently use another target's implementation. Reject unsupported
operations during resolution/lowering with target and capability details, not
through downstream source syntax or launch failures.

## Execution Backend Compatibility

A new target backend normally reuses an existing execution implementation. In
`tilelang/<backend>/execution_backend.py`, declare compatibility using
`ExecutionBackendSpec` rather than implementing target-local JIT/Runtime code.

For each compatible path, declare and verify:

- execution name and `auto` preference order;
- lazy availability check, when dependencies are optional;
- optional target predicate;
- whether the shared implementation requests host codegen;
- whether it requests eager device compilation;
- the compiled or source-only codegen mode it consumes;
- the source, artifact, global-symbol, argument, and launch-metadata contract.

Declaring `tvm_ffi`, for example, means the target backend can produce the host
and device runtime modules expected by the existing TVM FFI adapter. Declaring
`nvrtc` means it can produce CUDA source and metadata expected by the existing
NVRTC adapter. Do not declare an execution backend merely because its name is
already available.

The target backend provides target-specific source generation, validation,
compiler callbacks, and toolchain helpers. The shared execution backend owns:

- Build orchestration and cache policy;
- wrapper creation and artifact loading;
- argument, stream, and context binding;
- kernel launch, lifetime, and execution errors.

An execution implementation may call target-owned compiler callbacks during
Build. That compiler hook remains part of the target codegen/toolchain
contract; it does not require a target-specific JIT adapter.

Add a compatibility smoke test through each declared path, but do not duplicate
the shared adapter's generic unit tests in every target backend.

### When a new execution backend is required

Implement a new shared execution backend only when no existing adapter can
consume the generated artifact or operate the target runtime API. Then define:

- Build: compile, link, package, and artifact cache inputs;
- JIT: cache lookup, load, wrapper creation, and callable construction;
- Runtime: arguments, streams/contexts, launch, lifetime, and errors;
- AOT/export behavior when artifacts can be produced without loading.

`ExecutionBackendSpec` currently stores selection/capability metadata. Concrete
adapters live in `tilelang/jit/adapter/`, and execution-name switches remain in
shared JIT/cache/export/autotuning code. For a genuinely new mode, find every
site:

```bash
rg -n 'tvm_ffi|nvrtc|cython|torch|cutedsl' tilelang testing
```

Update JIT, cache, export, autotuning, diagnostics, environment validation, and
public type annotations as applicable. Keep bridge dispatch based on execution
name, never target kind. Do not place execution checks in compiler passes.

## Native Compilation and Packaging

Create `src/<backend>/CMakeLists.txt` and make it own source selection,
toolchain lookup, includes, definitions, stubs, and optional runtime libraries.

- Separate always-safe codegen/registration from SDK-gated runtime sources.
- Append sources/includes through existing `TILE_LANG_*` variables.
- Include the backend-local file from the root CMake delegation block.
- Add `USE_<BACKEND>` only when native compilation needs a gate; update
  `TILELANG_BACKENDS`, option docs, environment handling, and defaults together.
- Avoid hard runtime dependencies in wheels when codegen-only builds, lazy
  loading, or stubs are practical.
- Add link libraries, RPATH, install targets, and wheel-repair exclusions
  deliberately.
- Update `pyproject.toml` for dependencies, resources, or wheel behavior.

Python packages under `tilelang/` are included by the current wheel mapping.
Native sources are inactive until added through backend-local CMake.

Do not edit `3rdparty/` to bypass a missing TileLang integration point. Isolate
and justify any required upstream TVM target/runtime change.

## Tests and Documentation

Add hardware-independent coverage first:

1. Package import and manifest registration.
2. Target normalization and `create_backend_context()`.
3. Auto-detection if declared.
4. Compatible execution order, explicit selection, and unavailable errors.
5. Dialect exports and IR construction.
6. Minimal pipeline lowering plus every backend-specific IR form.
7. Source-only codegen without hardware when feasible.
8. Generated-source syntax/structure.
9. A smoke Build/load/launch through each declared shared execution path.
10. Full cache, wrapper, stream, argument, and lifetime tests when adding a new
    execution backend.
11. Unsupported operations/architectures.
12. Native build enabled and import when hardware/runtime is unavailable.

Update the builtin expectations in
`testing/python/backend/test_tilelang_backend_module.py`. Put target tests under
`testing/python/target/`, backend tests under `testing/python/<backend>/`, and
execution tests under `testing/python/jit/` or the backend directory. Add a
reusable marker under `tilelang/testing/` for hardware-gated tests.

Update:

- `tilelang/backend/README.md` registered-backend table;
- `docs/get_started/targets.md`;
- installation/build docs for SDKs and CMake flags;
- an example using the explicit dialect and target.

For exact build commands, use `.agents/skills/tilelang-build/SKILL.md`. Rebuild
after native edits to avoid testing a stale shared library. Typical focused
checks include:

```bash
python -m pytest testing/python/backend/test_tilelang_backend_module.py -x
python -m pytest testing/python/target/ -x
python -m pytest testing/python/foo/ -x  # Replace foo with the package name.
git diff --check
```

## Definition of Done

- [ ] Explicit dialect imports and constructs common and backend-specific IR.
- [ ] Target resolves to exactly one manifest and a compatible shared execution
      backend.
- [ ] Complete pass order lives in the backend package.
- [ ] Supported dialect IR lowers without engine target dispatch.
- [ ] Host/device codegen preserves source and launch metadata.
- [ ] Every declared shared execution backend can consume the generated output
      in a smoke test or reports an availability error.
- [ ] A target backend does not duplicate shared JIT/Runtime behavior.
- [ ] Common tile operations are implemented or fail explicitly.
- [ ] Native sources/toolchains are wired through local CMake and packaging.
- [ ] Hardware-free import/source tests run where feasible.
- [ ] Hardware tests cover launch and numerical correctness.
- [ ] Cache/export/autotuning behavior is covered for new execution modes.
- [ ] Registry, target docs, installation notes, and an example are updated.
- [ ] No backend-only symbol, pass order, or target dispatch leaked into shared
      code without demonstrated cross-backend need.
