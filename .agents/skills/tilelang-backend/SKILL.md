---
name: tilelang-backend
description: Integrate or review a TileLang target backend, target-backend variant, or execution backend end to end across language dialects, target and BackendContext resolution, PassPipeline, host/device codegen, Build/JIT/Runtime, native CMake and packaging, tests, and documentation. Use when asked to add, port, scaffold, complete, or audit a TileLang backend or execution mode.
---

# TileLang Backend Integration

Treat a target backend as one end-to-end vertical slice:

```text
Language Dialect
  -> BackendContext preparation
  -> PassPipeline
  -> HostCodegen + DeviceCodegen
  -> selected ExecutionBackend (Build -> Load -> Launch)
```

## Required Reading

Before editing code, read both files completely:

1. [`../../../tilelang/backend/README.md`](../../../tilelang/backend/README.md)
   for the architecture and ownership model.
2. [`references/integration-checklist.md`](references/integration-checklist.md)
   for concrete files, registration points, tests, and completion criteria.

When native code, installation, or tests are involved, also read
[`../tilelang-build/SKILL.md`](../tilelang-build/SKILL.md) before running build
commands.

## Workflow

### 1. Classify the Request

Choose exactly one primary category:

- **New target backend**: implement all five layers plus native/package
  integration, tests, and user documentation.
- **Target-backend variant**: add a distinct `BackendModule`; explicitly reuse
  or replace each component and make shared-target predicates unambiguous.
- **New execution backend**: implement one Build/JIT/Runtime path and declare it
  on compatible target backends; do not add a target pipeline unless target
  semantics change.

Do not conflate a backend name, Python package name, TVM target kind, and
execution backend name.

### 2. Define the Capability Contract

Before implementation, record:

- target kind, attributes, aliases, and variant predicate;
- supported dialect extensions and tile operations;
- launch/thread model and required pass behavior;
- device output format and host targets;
- execution backends in `auto` order;
- eager versus deferred device compilation;
- hardware-free and hardware-required validation boundaries.

If TVM does not recognize the target kind, treat TVM target registration as a
prerequisite. Do not emulate an unknown kind with scattered TileLang strings.

### 3. Inspect Before Copying

Use the closest existing backend as a reference:

- CPU for serial/non-SIMT lowering;
- ROCm, Metal, or WebGPU for a relatively direct SIMT backend;
- CUDA only when CUDA-specific scheduling and memory behavior are relevant;
- CuTeDSL for a predicate-selected variant sharing a target kind.

Trace package initialization, target normalization, pipeline, codegen global
functions, native source registration, execution adapter, CMake, and tests.
Preserve unrelated worktree changes.

### 4. Implement All Applicable Layers

Follow the detailed checklist in order. Keep these boundaries non-negotiable:

- Resolve target and execution state once into `BackendContext`.
- Keep backend language extensions outside `tilelang.language.common`.
- Keep the complete pass order in `tilelang/<backend>/pipeline.py`.
- Keep target dispatch out of `tilelang/engine/lower.py`.
- Keep target codegen and toolchain helpers under target-backend ownership.
- Keep Build inside the selected execution path.
- Add common helpers only after demonstrating a target-neutral shared use.
- Fail unsupported features during target resolution or lowering with an
  actionable message.

### 5. Reconcile Current Implementation Constraints

Account for these current transitional details:

- `BackendModule` is a compiler-facing manifest, not the entire backend.
- `ExecutionBackendSpec` describes selection and capabilities; concrete
  adapters still live under `tilelang/jit/adapter/`.
- Execution-name dispatch still exists outside the spec. Search all JIT, cache,
  export, autotuning, diagnostic, and public typing sites when adding a mode.
- Native `target.build.*` functions currently combine parts of CodeGen and
  Build. Preserve source-only codegen when deferred Build needs it.
- Backend Python packages are wheel-included automatically, but native sources
  require explicit backend-local CMake integration.

Do not deepen these transitional couplings. In particular, do not add
target-kind branches to execution dispatch or execution-name checks to passes.

### 6. Verify by Risk Layer

Run hardware-independent checks first:

1. package import and manifest registration;
2. target normalization and context resolution;
3. dialect exports and IR construction;
4. pipeline lowering and unsupported-feature diagnostics;
5. source-only codegen and generated-source assertions;
6. native build with the backend enabled;
7. execution Build/load/launch and numerical tests on target hardware;
8. cache/export/autotuning tests affected by a new execution mode;
9. `git diff --check` and relevant documentation checks.

After any native edit, rebuild before trusting Python results. Do not report a
backend complete when only registration or source generation passes.

## Completion Standard

Use the Definition of Done in the integration checklist. In the final report,
state:

- which of the five layers changed;
- which components were reused intentionally;
- supported and explicitly unsupported capabilities;
- execution paths tested;
- hardware/toolchain checks that could not be run.
