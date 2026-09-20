---
name: tilelang-semantic
description: Design, implement, debug, or review TileLang semantic rules and validation, including loop nesting, buffer bounds/initialization/ownership, data races, synchronization and async lifecycles, TileOp shape/dtype/scope contracts, software pipelines, reducers, layouts, kernel launch constraints, and actionable diagnostics. Use when changing tilelang/analysis checkers, early Verify* passes, language legality rules, or regression tests for invalid TileLang programs.
---

# TileLang Semantic Validation

## Core principle

Reject a program only when the violated semantic invariant is established.
Keep these outcomes distinct:

| Finding | Treatment |
|---|---|
| Semantically invalid for every supported lowering | Default error |
| Valid in principle but unsupported by the current lowering | Precise unsupported-case diagnostic or fallback |
| Valid but an optimization does not apply | Warning or silent fallback, never a semantic error |
| Potentially invalid but not provable | Warning, opt-in verifier, or no report |

Do not turn an optimizer limitation into a language rule. For example,
`T.vectorized` containing a loop-invariant `T.serial` is semantically valid even
though the current vectorization planner may scalarize it.

## Workflow

### 1. Reconstruct the actual contract

Before proposing or changing a rule:

1. Locate the source API under `tilelang/language/`.
2. Determine its source TIR representation. Do not infer semantics from the
   Python spelling alone: `T.Pipelined` is a serial `For` with annotations, and
   `T.Persistent` expands early into binds plus a serial loop.
3. Trace the construct through the target pipelines and identify the first pass
   that assumes the proposed invariant.
4. Search tests and examples for valid counterexamples.
5. Load the relevant domain reference completely:

   - loop nesting and loop/storage ownership:
     `references/loop-rules.md`;
   - bounds, initialization, scopes, races, synchronization, barriers, or
     async operations: `references/memory-concurrency.md`;
   - TileOps, software pipelines, reducers, layouts, kernel launch, or dtype
     contracts: `references/operations-layout.md`.

For loop-nesting work, also run:

```bash
python .agents/skills/tilelang-semantic/scripts/scan_loop_nesting.py
```

Absence from the repository is not proof that a construct is invalid. Confirm
the downstream assumption in code or with a minimal lowering test.
Treat the scanner as an inventory aid, not a proof: inspect non-literal
annotation dictionaries and generated TIR manually.

### 2. Specify the rule before implementing it

Write down all of the following:

- **Scope:** the exact node, storage scope, or lexical region covered.
- **Invariant:** what every valid program must satisfy.
- **Proof criterion:** what the checker must establish before reporting.
- **Exemptions:** atomics, reducers, generated IR, target-specific paths, etc.
- **Near-neighbor valid case:** the smallest similar program that must remain
  accepted.
- **Diagnostic:** name the offending construct and give an actionable rewrite.
- **Enforcement stage:** frontend, pre-lower, early native verification,
  post-inference, or backend-specific lowering.

For nesting rules, define a *path* as one lexical ancestor chain from the
function body to a leaf statement. Sequential sibling loops are different
paths.

### 3. Choose the earliest reliable enforcement stage

| Stage | Use it for |
|---|---|
| Frontend API | Invalid argument combinations or types known while constructing the loop/op |
| `PreLowerSemanticCheck` | Backend-independent source-TIR structure with user-facing syntax still recognizable |
| Early native `Verify*` pass | Analyzer-, effect-, region-, or dataflow-dependent checks shared by backends |
| After `LayoutInference` | Contracts involving inferred `layout_map` or `parallel_loop_layout` |
| Backend pipeline | Rules that genuinely depend on target instructions or execution geometry |

Do not place a rule after a transform that erases the evidence needed to report
it. If an early expansion erases identity, preserve an explicit annotation
instead of pattern-matching incidental lowered shapes.

### 4. Implement a validator, not a transform

- Keep validation side-effect free.
- Maintain lexical state with a stack and restore it on every exit path.
- Use `For`, `Buffer`, `Var`, and other ObjectRef handles for retained identity;
  follow `$tilelang-tvm-ir` for C++ TIR code.
- Share traversal/variable-use utilities when rules need the same facts, but
  keep diagnostics and rule ownership independent.
- Include source spans when the IR carries them.
- Use a stable prefix such as `[TileLang Semantic Check]` and describe a legal
  replacement.
- Honor the existing global pre-lower opt-out for compatibility. Do not add a
  new per-rule opt-out for a proven-invalid rule unless compatibility requires
  one; reserve fine-grained opt-outs for inconclusive analyses.

### 5. Test the semantic boundary

Add, at minimum:

1. one failing test for each violation shape;
2. one near-neighbor valid test;
3. a proof-boundary case that remains accepted when the analysis is
   inconclusive;
4. alternate control-flow paths for state/lifecycle rules;
5. target-specific accepted and unsupported cases when capability matters;
6. a diagnostic assertion that checks the useful part of the message.

Also run affected backend codegen tests when a rule migrates existing kernels.
Use `$tilelang-build` for repository test commands. For a Python pre-lower
checker, start with:

```bash
python -m pytest testing/python/analysis -q
python -m ruff check <changed-python-files>
git diff --check
```

## Validation ownership map

- `tilelang/engine/semantic_check.py`: shared pre-lowering entry point.
- `tilelang/analysis/*checker.py`: source-TIR structural rules and actionable
  frontend diagnostics.
- `src/transform/verify_*.cc`: native effect/dataflow verification such as
  reducer lifecycle, buffer initialization, and parallel races.
- `src/transform/legalize_safe_memory_access.cc`: bounds proof, global guards,
  and optional local/shared warnings.
- `tilelang/language/*_op.py` and `tilelang/language/builtin.py`: source API
  operand and annotation validation.
- `src/transform/{pipeline_planning,inject_pipeline}.cc`: pipeline dependency,
  ordering, replayability, and multi-versioning contracts.
- `src/transform/layout_inference/parallel_loop_layout_validator.h`:
  post-inference parallel-layout annotation contract.
- `tilelang/{cuda,rocm,cpu,metal,webgpu}/pipeline.py`: target pipeline order.
- `tilelang/transform/pass_config.py`: opt-outs and strictness controls.

Read `$tilelang-layout` before changing rules whose truth depends on fragment
ownership, replication, loop partitioning, or inferred loop layouts.

## Avoid

- Do not equate `T.Parallel` with `T.vectorized`; they represent different
  execution layers and use different lowering paths.
- Do not reject every construct that falls back to serial execution.
- Do not promote “cannot prove safe” to “proven unsafe.”
- Do not use Python `assert` for new user-facing legality checks; assertions
  disappear under optimized Python and usually produce poor diagnostics.
- Do not infer source intent from `ForKind` alone when annotations define the
  construct.
- Do not treat nested pipeline requests as backend-independent invalidity;
  hierarchical-pipeline support is a backend capability.
- Do not require parallel extent to equal thread count or explicit layout
  shape; partitioning, replication, and guarded tails intentionally permit
  differences.
- Do not reject barriers merely because they occur inside `T.Parallel`;
  validate participant sets, path uniformity, and execution counts.
- Do not report a missing target feature or missed optimization as universal
  semantic invalidity.

## Resources

- `references/loop-rules.md`: established loop contracts, representation
  details, non-rules, and backend-specific nested-pipeline guidance.
- `references/memory-concurrency.md`: buffer bounds/initialization/ownership,
  aliasing, races, collective synchronization, barriers, and async lifetimes.
- `references/operations-layout.md`: TileOp, pipeline, reducer, layout,
  kernel/target, and dtype contracts plus the implementation roadmap.
- `scripts/scan_loop_nesting.py`: inventory lexical loop pairs and detect
  nested software-pipeline requests in Python sources without judging backend
  support.
