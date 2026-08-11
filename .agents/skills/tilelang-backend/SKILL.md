---
name: tilelang-backend
description: Integrate or review a TileLang target backend, target-backend variant, or shared execution backend. Covers language dialects, target and BackendContext contributions, PassPipeline, host/device codegen, execution compatibility, native CMake and packaging, tests, and documentation; covers Build/JIT/Runtime implementation only when a genuinely new execution mode is requested. Use when asked to add, port, scaffold, complete, or audit a TileLang backend or execution mode.
---

# TileLang Backend Integration

Treat a target backend as four compiler-owned implementation areas connected to
a separately reusable execution backend:

```text
Language Dialect
  -> target/BackendContext contribution
  -> PassPipeline
  -> HostCodegen + DeviceCodegen
  -> compatibility contract

BackendContext selects a shared ExecutionBackend
  -> Build -> Load -> Launch
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

- **New target backend**: implement the four target-backend areas, declare
  compatibility with existing execution backends, and add native/package
  integration, tests, and user documentation.
- **Target-backend variant**: add a distinct `BackendModule`; explicitly reuse
  or replace each component and make shared-target predicates unambiguous.
- **New execution backend**: only when no existing implementation can consume
  the artifact or runtime API, implement one shared Build/JIT/Runtime path and
  declare compatible target backends; do not add a target pipeline unless
  target semantics change.

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
- Declare execution compatibility; do not implement target-local JIT/Runtime
  when an existing shared execution backend satisfies the contract.
- Add common helpers only after demonstrating a target-neutral shared use.
- Fail unsupported features during target resolution or lowering with an
  actionable message.

### 5. Reconcile Current Implementation Constraints

Account for these current transitional details:

- `BackendModule` is a compiler-facing manifest, not the entire backend.
- `ExecutionBackendSpec` describes selection and capabilities; concrete
  adapters still live under `tilelang/jit/adapter/`.
- For a new target backend, supplying compatible codegen outputs and metadata
  is normally sufficient; a new adapter is not.
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
7. a compatibility smoke test through each declared shared execution path;
8. full Build/load/launch, cache, export, and autotuning tests only for a new
   execution mode;
9. `git diff --check` and relevant documentation checks.

After any native edit, rebuild before trusting Python results. Do not report a
backend complete when only registration or source generation passes.

## Completion Standard

Use the Definition of Done in the integration checklist. In the final report,
state:

- which of the four target-backend areas changed;
- which components were reused intentionally;
- supported and explicitly unsupported capabilities;
- compatible shared execution paths tested;
- hardware/toolchain checks that could not be run.
