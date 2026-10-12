# Testing Style Guide

Use this guide when designing or changing tests in `testing/`, tests in
`examples/`, and their shared helpers. Preserve necessary regression protection
with small maintenance, compilation, and execution costs. The goal is not the
fewest test functions or a fixed number of cases per pull request.

## Decide What Coverage Is Missing

Start with the changed behavior and the closest existing tests, before writing a
new reproducer into the suite. Check their inputs, assertions, backend, and pass
configuration: a nearby helper may disable the very transformation being fixed.

| Situation | Preferred change |
| --- | --- |
| Existing tests already protect the affected behavior and trigger | Run that coverage; do not add an equivalent permanent test just to accompany the fix. |
| An existing test covers the same contract but misses the triggering input or assertion | Add the smallest necessary case or assertion while preserving its existing coverage. |
| Extending a test would multiply unrelated parameters or obscure a different contract | Add a focused test, reusing suitable helpers rather than duplicating the entire setup. |
| No existing family can reasonably express the behavior | Add a focused test in the appropriate semantic or backend area. |

A passing suite or overlap in executed source lines does not establish equivalent
coverage. For a regression, identify the error the retained test would detect;
when practical, verify that it fails for that reason before the fix and passes
afterward. Distinguish unverified reasoning from an actual reproduction.

A temporary debugging script need not become another permanent test if the
maintained suite already protects the same failure. Do not use a temporary
reproduction as a substitute for missing regression coverage, or remove a unique
case merely because it resembles another test.

## Find the Existing Test Home

Search the relevant family and its shared helpers rather than starting a new
issue-specific file. These are entry points, not templates to copy wholesale:

| Behavior | Look first |
| --- | --- |
| Language operations such as copy and reduction | `testing/python/language/` |
| Compiler transformations, analysis, or layouts | `testing/python/transform/`, `testing/python/analysis/`, `testing/python/layout/` |
| Generated source or backend-specific behavior | `testing/python/target/` and the relevant backend directory under `testing/python/` |
| JIT, runtime, or kernel integration | `testing/python/jit/`, `testing/python/runtime/`, `testing/python/kernel/` |
| An example operator's end-to-end contract | The corresponding test under `examples/` |
| Ascend-specific behavior | `testing/ascend/` and its existing `README.md` |

Existing files under `testing/python/issue/` do not mean every bug fix needs a new
issue file. Prefer a name and location that describe the behavior. Keep separate
functions when they make different contracts clearer; do not combine unrelated
tests simply to reduce file or function counts.

## Choose the Smallest Sufficient Test Layer

- For behavior observable in language construction or a compiler pass, prefer a
  focused language or IR check over a full operator execution.
- For a code-generation contract, inspect the relevant generated instruction or
  property with an explicit target. Prefer structural IR checks or focused
  source assertions over snapshots of entire generated programs.
- For wrong numerical results, memory safety, synchronization, or another
  execution-dependent failure, retain the runtime test needed to expose it.
  A source-generation check is not a replacement for that evidence.
- Compilation-only tests can still require a backend to be built and registered.
  Apply the appropriate backend requirement, such as
  `tilelang.testing.requires_cuda`, even when no kernel runs. Preserve physical
  device capability requirements when execution depends on them.

Assert the intended result, diagnostic, IR property, or generated-code contract.
Do not read repository source and assert that the text of the fix is present.
Reuse an appropriate independent reference or existing verifier rather than
deriving the expected result from the implementation under test.

## Select Inputs and Parameters Deliberately

- Start with the smallest legal input that preserves the failure. Keep required
  alignment, layout, tile size, pipeline state, or architecture constraints;
  minimizing away the trigger does not produce a regression test.
- Add a shape, dtype, stage count, or thread count because it protects a distinct
  affected path or boundary, not merely because it is another supported value.
  Retain combinations when the failure depends on their interaction.
- Prefer explicit relevant combinations over a Cartesian product of unrelated
  parameters. This is not a one-case limit: multiple affected behaviors need
  their own protection.
- Inspect inherited parameterization, fixtures, and loops before adding a case.
  One more shape can expand across every dtype and configuration, and one test
  function can still compile or execute many kernels.
- Keep performance sweeps and large stress workloads out of a small correctness
  regression unless their scale is necessary to reproduce the failure. Use the
  repository's existing benchmark or stress-test locations for those purposes.

## Reuse Without Building Another Test Framework

Reuse nearby kernel builders, references, and verification helpers when they
exercise the required behavior. A new test usually should not copy an entire
kernel and reference implementation only to change one shape. Extract shared
setup when it has a concrete use, but do not introduce a broad fixture framework
or refactor unrelated tests to accommodate a small fix.

## Examples of Choosing Scope

### Copy Tail Handling

Start with `testing/python/language/test_tilelang_language_copy.py` and its
`run_tilelang_copy` helper. Confirm that its pass configuration reaches the bug.
If the failure is a row tail beyond a tile boundary, preserve the existing aligned
coverage and add a minimal legal row-tail input to a suitable test.

For example, `rows = block_rows + 1` is a candidate when one extra row is the
trigger; it is not a universal test size. Keep any dtype or configuration needed
to reproduce the fault. Avoid creating another full copy kernel and sweeping
unrelated shapes, stage counts, and thread counts for this one failure.

### Simplification of an Index Expression

Start with `testing/python/transform/test_tilelang_transform_simplify.py` and
`simplify_and_compare`. Express the smallest before/expected IR pair, including
the assumptions needed to trigger the incorrect rewrite.

If that comparison exposes the bug, another end-to-end GEMM is unnecessary.
Conversely, do not replace an execution-only failure with an IR assertion that
cannot distinguish the broken behavior.

### Backend-Specific Code Generation

`testing/python/target/test_tilelang_codegen_clamp.py` illustrates explicit
targets and per-case backend requirements. Different generated-code contracts
can justify separate backend cases even when they originate from the same
language operation. Add coverage for the affected backend or shared behavior,
not an unrelated matrix of architectures and launch configurations.

## Validate the Chosen Scope

Run the focused test and the relevant existing coverage for the changed behavior.
Report what actually ran and any unavailable backend or toolchain. Changes to
shared helpers need validation of their affected callers; a small edit to shared
setup is not necessarily a small coverage change. Documentation-only changes do
not require inventing runtime tests.
