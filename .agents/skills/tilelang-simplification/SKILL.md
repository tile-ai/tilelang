---
name: tilelang-simplification
description: Find, evaluate, propose, implement, or review non-obvious simplifications in TileLang. Use for broad simplification audits; dead or duplicated compiler, runtime, backend, layout, build, test, or API surface; collapsing unnecessary abstractions or pass machinery; removing stale compatibility paths; or replacing hand-rolled infrastructure. Do not use for a routine localized refactor whose desired change is already known.
---

# TileLang Simplification

Turn a broad request to "simplify TileLang" into a small set of
evidence-backed changes that reduce maintained concepts, contracts, or code
paths. Fewer lines are not sufficient: preserve correctness, useful
diagnostics, supported targets, public compatibility, compile behavior, and
kernel performance unless the user explicitly accepts a tradeoff.

## Choose The Working Mode

- **Survey**: inspect broadly and report ranked candidates. Do not modify code,
  add TODOs, or create issues unless requested.
- **Implement**: prove the selected candidate first, then make the smallest
  complete change, including tests, documentation, exports, registration, and
  build metadata affected by the removal.
- **Review**: determine whether a proposed simplification is a net reduction
  and whether its evidence and validation cover all affected targets.

For a broad survey, continue past the first plausible candidate. Prefer a few
high-confidence reductions over a long list of speculative cleanups.

## Establish Repository Context

Read `README.md` and `CONTRIBUTING.md`, inspect the worktree, and identify the
requested base revision before judging current code. Exclude `3rdparty/`,
`build/`, `docs/_build/`, caches, generated files, and vendored target stubs by
default; include them only when the request specifically concerns generated,
vendored, or packaging surface.

Load the relevant domain skill before judging a candidate in that area:

- target or execution backends: [`../tilelang-backend/SKILL.md`](../tilelang-backend/SKILL.md)
  and `tilelang/backend/README.md`;
- layouts, fragments, inference, or CuTe conversion:
  [`../tilelang-layout/SKILL.md`](../tilelang-layout/SKILL.md);
- legality checks, safety checks, or diagnostics:
  [`../tilelang-semantic/SKILL.md`](../tilelang-semantic/SKILL.md);
- C++ TIR/ObjectRef ownership:
  [`../tilelang-tvm-ir/SKILL.md`](../tilelang-tvm-ir/SKILL.md);
- broad C++ cleanup: [`../tilelang-cpp-style/SKILL.md`](../tilelang-cpp-style/SKILL.md);
- builds and tests: [`../tilelang-build/SKILL.md`](../tilelang-build/SKILL.md).

Do not load every domain skill for a narrow audit.

## What Counts As A Strong Candidate

A strong candidate removes or folds a real maintenance burden and has direct
evidence that the current surface costs more than it protects. Typical
candidates include:

- an internal API, option, pass, callback, registry entry, helper, or package
  with no production consumer;
- a Python wrapper, C++ entry point, or FFI binding that duplicates another
  representation without owning validation, compatibility, or policy;
- target selection or execution dispatch repeated outside `BackendContext` or
  a backend-owned pipeline;
- backend-local implementations with genuinely identical semantics that can
  move behind an existing common boundary;
- pass configuration, annotations, analyses, or state that are produced and
  propagated but never observed;
- adjacent passes or lifecycle mechanisms that preserve the same fact and can
  be represented once;
- obsolete compatibility facades, aliases, or migrations whose supported
  callers and documented transition window are gone;
- tests, snapshots, examples, or generated expectations that exist only to pin
  an unused implementation detail;
- hand-rolled utilities already provided by the Python/C++ standard library,
  TVM, or an existing dependency, when replacement deletes the implementation
  and its special-case tests without hiding equivalent complexity in glue;
- build, packaging, or toolchain branches for artifacts that are no longer
  produced or consumed.

Weak candidates include style-only churn, a single typo, an abstraction that
merely looks complicated, or a target-specific path that appears unused only
because the current machine cannot exercise it. Moving the same complexity
behind a new wrapper is not simplification.

## Survey By Ownership Area

For repository-wide work, divide the survey into these areas, in parallel when
practical:

1. language APIs, Python exports, IR objects, FFI registrations, and
   compatibility aliases;
2. shared transforms, pass configuration, analyses, annotations, and pipeline
   ordering;
3. target backends, execution backends, codegen, runtime, cache, and toolchain
   integration;
4. layouts, intrinsics, MMA/TMA paths, vectorization, and architecture-specific
   lowering;
5. autotuning, profiler, carver, diagnostics, instrumentation, and developer
   tooling;
6. CMake, packaging, tests, examples, benchmarks, maintenance scripts, and
   documentation.

Start with large or repeated production-code surfaces, not only symbols found
by an unused-code tool.

## Prove Consumers And Reachability

Use `rg` first, then read every relevant call site. Search the exact C++ or
Python symbol, its exported alias, registered string, pass-config key, target
name, annotation key, and generated symbol. Classify references as:

- **production**: `tilelang/`, `src/`, `cmake/`, `CMakeLists.txt`, packaging
  metadata, runtime templates, and loader or registration paths;
- **coverage and usage evidence**: `testing/`, `examples/`, `benchmark/`, and
  `maint/`; examples and maintenance programs may be supported smoke paths,
  so inspect them before dismissing them;
- **documentation only**: `README.md` and `docs/`;
- **excluded/generated/vendor**: the excluded trees listed above.

Static text search is not sufficient for TileLang. Before declaring code dead,
check all applicable indirect-use mechanisms:

- `TVM_FFI_STATIC_INIT_BLOCK`, `TVM_REGISTER_*`,
  `TIR_REGISTER_TL_TILE_OP`, reflected object fields, and global function
  names;
- Python `tvm_ffi.register_object`, `_ffi_api` bindings, `__init__.py`
  re-exports, `__all__`, decorators, and compatibility facades;
- `BackendModule` manifests, target predicates, execution compatibility,
  callback maps, and backend package import side effects;
- pass factories referenced by registered names, backend pipeline lists, and
  `PassContext` configuration strings;
- CMake source lists and object files retained for static initialization;
- generated runtime-template symbol names, cache keys, serialized artifacts,
  and codegen-emitted calls.

Useful discovery searches include:

```bash
rg -n "ExactSymbol|registered.string|config_key" tilelang src cmake \
  CMakeLists.txt testing examples benchmark maint docs
rg -n "TVM_FFI_STATIC_INIT_BLOCK|TVM_REGISTER_|TIR_REGISTER_TL_TILE_OP|register_object|register_backend|__all__" \
  tilelang src
```

Treat static analyzers as inventory aids, not proof; dynamic FFI, registration,
and target dispatch routinely hide valid consumers.

Absence of an in-tree caller proves only in-tree reachability. Treat exported
Python APIs, FFI and reflection names, pass-config and annotation strings,
target names, runtime ABI, and cache or serialized formats as externally
consumed until documentation, release history, and deprecation policy provide
evidence otherwise. Tests describe important cases but are not the sole source
of the contract.

## Trace The Compiler Contract

For a compiler or language candidate, trace one representative construct from
the source API through its TIR/TileOp representation, annotations, selected
backend pipeline, lowering passes, host/device split, codegen, and runtime
consumption. Name the first stage that requires the surface being considered.

Before merging or deleting passes, determine:

- which backend pipelines include each pass and in what order;
- the preconditions, annotations, and analyses each pass consumes and emits;
- whether a pass is a no-op only for the inspected example or architecture;
- whether diagnostics or source spans become less useful after moving it;
- whether generated source, launch metadata, cache identity, or runtime ABI
  changes even when final numerical results do not.

Do not replace a semantic check with an optimizer assumption, or delete a
fallback merely because the optimized path handles common examples.

When a new dependency would replace hand-rolled code, prefer the standard
library, TVM, or an existing dependency first. Otherwise check supported
Python versions and platforms, wheel or native-build availability, license,
maintenance, transitive footprint, and net code deletion. A dependency that
moves equivalent complexity into adapters, CMake, or wheel packaging is not a
simplification.

## Respect Backend And Hardware Boundaries

Target backends and execution backends are intentionally separate. Do not
collapse CUDA, CuTeDSL, ROCm, CPU, Metal, WebGPU, or their execution adapters
solely because implementations currently resemble one another. A shared
helper is justified only when ownership is target-neutral and the supported
capability differences remain explicit.

Absence of a runnable device is not evidence that a path is dead. For every
backend-facing candidate, record:

- target kinds and architecture predicates that can reach it;
- source-only or hardware-independent checks available locally;
- hardware and toolchain coverage that would still be required;
- intentional unsupported cases and fallback behavior.

## Protect Performance-Sensitive Semantics

A simpler compiler implementation can produce slower kernels or slower
compilation while remaining numerically correct. Scale evidence to the risk:

- for cold Python helpers, imports and focused unit tests may be enough;
- for transforms and codegen, compare relevant IR and generated source;
- for layout, vectorization, memory movement, synchronization, MMA/TMA,
  pipeline, or autotuning changes, run the subsystem harness and representative
  hardware benchmarks when available;
- for build, cache, JIT, or runtime changes, verify Build, load, launch, cache
  hit behavior, and artifact portability that the path claims to support.

If hardware validation is unavailable, report the gap. Do not convert missing
performance evidence into a claim of equivalence.

## Decide The Candidate

Classify each investigated item:

- **safe deletion**: internal, unreachable through direct or indirect paths,
  and not part of a supported compatibility or artifact contract;
- **consolidation**: multiple paths implement the same contract and an existing
  owner can absorb them with less total branching and state;
- **behavior or API tradeoff**: simplification changes a public API, FFI name,
  IR/annotation contract, target capability, generated source, cache format, or
  performance characteristic; require explicit acceptance;
- **unproven or intentional**: keep it and record why the apparent duplication
  is load-bearing.

Use history to understand intent when current code is ambiguous, but judge
reachability and contracts from the current tree. Reject a candidate when the
proposal forces broad unrelated churn without reducing concepts, ownership
boundaries, or supported behavior.

## Report Or Implement

For a survey, report each retained candidate with:

1. affected files, symbols, or registered names;
2. current responsibility and production consumers;
3. exact removal or consolidation proposed;
4. compatibility, target, correctness, and performance tradeoffs;
5. validation required and any unavailable hardware;
6. confidence level.

Also mention representative rejected candidates when they explain intentional
architecture or prevent repeated false positives.

For implementation, remove the whole obsolete surface: definitions,
declarations, exports, registrations, configuration, CMake/package entries,
tests that only pin deleted behavior, and stale documentation. Preserve tests
that express the surviving contract, and add a regression test when the new
boundary would otherwise be ambiguous. Use the repository formatter on touched
files, follow [`../tilelang-build/SKILL.md`](../tilelang-build/SKILL.md) for
build and test commands, and always run `git diff --check`.

In the final report, state the conceptual surface removed, the supported
behavior retained or intentionally given up, the checks run, and hardware or
toolchain validation not performed.
