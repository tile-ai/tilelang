---
name: tilelang-op-test
description: Design, generate, supplement, and accept trustworthy system tests for operators in the current repository. Use when delivering operator ST from scratch based on requirements or a design, addressing existing pytest suites and their coverage gaps, or performing delivery acceptance on CUDA/Ascend. Covers functionality, correctness, boundaries, gradients, state mutation, invalid-input rejection, layout and interfaces, backend paths, randomness, and execution evidence.
license: MIT
metadata:
  audience: Operator developers
  workflow: System testing
---

# System Testing for the Current Repository

Use one Skill entry point to provide trustworthy evidence that an operator satisfies its contract in the target environment. The maintainable long-term asset is executable pytest with an independent oracle and traceability to the contract; the artifact tied to a particular run is the acceptance report.

Trustworthiness consists of three indispensable parts:

```text
Trustworthy cases × Complete applicable coverage × Successful execution evidence
```

Complete coverage reduces untested behavior and therefore increases confidence, but it cannot compensate for a fabricated contract, an oracle that depends on the implementation under test, invalid inputs, or weak assertions. Count only trustworthy verification conclusions when calculating coverage.

Before using scripts bundled with this Skill, set `ST_SKILL_DIR` to the actual directory containing the current `SKILL.md`, then locate `scripts/` and `references/` from that directory using their internal relative paths. Do not depend on the host tool's installation path.

## Select a Business Scenario

- **Delivery from scratch**: Begin with user requirements, an operator prototype, interface documentation, or a design. Produce a contract checklist, independent oracle, complete applicable coverage design, executable pytest, and acceptance evidence. A kernel or test file does not need to exist when test design begins.
- **Existing-suite improvement**: Inspect repository code and tests, determine whether existing tests are trustworthy, identify missing dimensions, and add or repair the smallest effective set of pytest cases.
- **Delivery acceptance**: Bind a `PASS`, `FAIL`, or `NOT_VERIFIED` conclusion to a fixed commit/worktree, specified backend and device, trustworthy coverage, and recorded execution results.

For delivery from scratch or a version release, first read [Delivery Workflows](references/delivery-workflows.md). Respect a smaller user-specified scope: when asked only to review or design, produce review materials without modifying tests; a request to add, complete, or deliver ST authorizes implementing and validating pytest.

## Mandatory Workflow

### 1. Locate or Define the Public Interface and Test Entry Point

For an existing operator, record its public function, dispatch wrappers, kernels for each backend, independent reference, existing tests, and target device/backend. For delivery from scratch, determine the agreed public signature and supported scope from the requirements or design before writing the reference or tests.

Identify pytest entry points from actual files in the current repository:

- Framework tests under `testing/ascend/`, `examples/ascend/test_*.py`, and corresponding tests in each example subdirectory;
- User-specified tests, `test_*.py` files in operator artifact directories, and embedded `test_*` functions in example files. Embedded tests usually require an explicit file or nodeid.

When the target is not yet clear, run the static discovery tool before importing TileLang or torch:

```bash
python "$ST_SKILL_DIR/scripts/discover_tests.py" \
  --repo-root . --symbol <public_function> --format markdown
```

Before collecting or running tests, read [Repository Test Execution](references/repository-testing.md), especially when the work involves source-embedded tests, NPU execution, xdist, or test levels.

### 2. Establish the Contract Before Assessing Coverage

Build a concise list of requirements and their sources, using the following evidence priority:

1. User-confirmed requirements or authoritative interface/design documentation;
2. Public API signatures and docstrings;
3. An independent PyTorch/CPU reference or explicit mathematical definition;
4. Mature equivalent CUDA/legacy implementation behavior as supporting evidence;
5. Existing tests and implementation inspection as evidence of current behavior.

Use the implementation to identify dispatch and boundary paths. Do not use only the implementation under test to prove expected semantics. Explicitly identify conflicts between sources. When required behavior has no reliable source, mark it `CONTRACT_GAP`; do not invent pass/fail assertions.

When reviewing or creating correctness assertions, read [Test Credibility](references/test-credibility.md).

### 3. Review Trustworthiness Before Counting Coverage

Inspect every relevant test:

- Whether a positive case's input is valid for the scenario it verifies, and whether a negative case has an explicit invalid-input rejection contract;
- Whether the oracle is sufficiently independent for the current verification conclusion;
- Whether assertions check every material output, gradient, or state change;
- Whether the case actually enters the claimed backend and path;
- Whether pytest collects and executes the test, rather than it merely existing in source;
- Whether the result and actual random seed used are reproducible.

An assertion that checks only shape proves only shape, not numerical correctness. A differential comparison against CUDA alone is only supporting evidence; primary semantic conclusions require a specification, independent reference, explicit expected values, or a well-justified metamorphic relation.

Mark every verification conclusion separately as `TRUSTED`, `PARTIAL`, `UNTRUSTED`, or `UNKNOWN`. The same test may be trustworthy for output shape but only partially trustworthy for numerical values.

### 4. Evaluate Ten Coverage and Acceptance Dimensions

For every dimension, record `covered`, `partial`, `missing`, `unknown`, or `not_applicable`; link the corresponding case IDs and explain the rationale. `not_applicable` must be a reviewed conclusion, not an empty default.

| No. | Dimension | Required Checks When Applicable |
|---:|---|---|
| 1 | Functionality | Formula, masks, broadcasting, optional behavior, multiple outputs |
| 2 | Correctness | Exact comparison, or justified tolerance by output/dtype |
| 3 | Boundaries | Minimums, empty inputs, valid extremes, and genuine tile/core tails |
| 4 | Gradients | Every required backward result compared with an autograd-capable reference |
| 5 | State Mutation | Changes that must occur, and inputs, regions, padding, or storage that must remain unchanged |
| 6 | Invalid-Input Rejection | Documented invalid inputs, exact exception type, and stable error reason |
| 7 | Layout and Interface | Shape, dtype, device, stride, contiguity, alias relationships, metadata |
| 8 | Backend and Branches | Required CUDA/Ascend dispatch paths and key algorithmic paths |
| 9 | Randomness | Random-seed semantics, reproducibility, invariants, and distributions for stochastic operators |
| 10 | Execution Evidence | Test collection, nodeids, target runtime, results, environment, and random-seed strategy |

The first nine dimensions describe operator and test coverage; execution evidence shows whether the designed coverage actually ran. Record these additional classification axes separately:

- Repository test level: `0` core, `1` default, `2` full;
- Test kind: ordinary, boundary, negative, gradient, stateful, benchmark;
- Contract dimension: dtype, rank/shape, attributes, optional-parameter dependencies, layout/stride, dispatch/backend, numerical range, side effects, outputs, and gradients.

Derive boundary candidates from the public valid domain and actual dispatch and tiling code. `B+1` is a valid positive tail case only if the public contract allows that input and the final runtime dimension genuinely has a remainder. Before adding boundary, negative, gradient, or stateful cases, read [Coverage Design](references/coverage-design.md).

Do not target a fixed case count. Prefer a compact set of cases that distinguishes meaningful paths and their interactions while avoiding unnecessary JIT variants and memory cost.

### 5. Implement Maintainable ST Assets

- For delivery from scratch, create a separate test file and a reviewable PyTorch/CPU reference if they do not already exist. Tests must call the agreed public API and must not implement the production kernel inside the test.
- For existing-suite improvement, extend an existing test file when it provides suitable fixtures and conftest behavior.
- Reuse `get_device()`, `get_test_level()`, generators, references, `make_param_id`, and numerical helpers when their semantics match.
- Give new parameterized cases stable, descriptive IDs.
- Preserve existing reference logic and tolerances unless an authoritative contract proves they must change.
- Compare every material output. For an in-place kernel, clone DUT and reference inputs separately, and check protected storage or unchanged regions when applicable.
- Negative cases must check the explicit expected exception. Unrelated import, device, compilation, or OOM exceptions do not count as successful rejection tests.
- Pytest must fail when a valid boundary produces an incorrect result; do not downgrade it to a warning.

Test designs for delivery from scratch and delivery acceptance use the JSON format defined in [Case Specification](references/case-specification.md). It may be omitted only for a change to a small existing operator when the contract mapping is exceptionally clear. Validate it with:

```bash
python "$ST_SKILL_DIR/scripts/check_st_spec.py" \
  path/to/spec.json --repo-root .
```

Add `--require-complete-coverage` for final acceptance. At that point, any `partial`, `missing`, or `unknown` dimension blocks `PASS`.

### 6. Collect, Run, and Preserve Execution Evidence

First collect exact targets in the same backend and device environment used for formal execution. A static `node_hint` may only discover test functions; backend-dependent generators such as `is_ascend()` may produce different parameterized nodeids. Select concrete nodeids from collection results in the target environment, then validate them without xdist before a batch run. A zero collection count or all relevant tests being skipped yields `NOT_VERIFIED`, never a pass. Run new or modified cases first, followed by the established complete target-case list. When broader coverage is required, run expanded tests according to the current project's actual test configuration.

Accept PTO or AscendC as the target backend at invocation. For test collection, new cases, and final full correctness acceptance, explicitly specify the corresponding backend before each command: use `TILELANG_DEFAULT_TARGET=pto` for PTO and `TILELANG_DEFAULT_TARGET=ascend` for AscendC. Commands for the execution-evidence script below use the same prefix. After a fix, rerun final acceptance on the selected backend.

For ordinary standalone tests, command arguments and device concurrency follow this Skill's [Test Execution Guide](references/repository-testing.md); the ST final-acceptance scope follows the preceding paragraph. Explicitly select and separately run source-embedded tests. Do not refresh benchmark baselines or memory profiles unless the user requests it.

Record the commit/diff, environment, actual imported dependency paths, target backend/device, complete commands, collected nodeids, result counts, failure/skip information, and actual random-seed strategy. Clearly distinguish test failures, collection errors, dependency errors, compilation errors, device-resource errors, interruption, and timeout.

When a persistent JSON record is needed, use the execution-evidence recording tool:

```bash
python "$ST_SKILL_DIR/scripts/pytest_st_evidence.py" \
  --repo-root . --output <artifact-dir>/pytest-evidence.json \
  --device "Ascend NPU" --backend "PTO/Ascend" -- \
  <test_file> -q
```

This tool proves only facts about test collection and execution. Contract validation and trustworthiness review must also be complete before assigning `PASS`.

### 7. Give the Delivery Conclusion

State the correctness conclusion and its scope first, including:

| Item | Required Content |
|---|---|
| Contract | Requirements, sources, backend, and degree of certainty |
| Trustworthiness | Trusted, partially trusted, untrusted, and unknown verification conclusions with reasons |
| Coverage | Dimensions with trustworthy coverage and remaining gaps |
| Changes | New cases and the independent defect class each case can detect |
| Execution | Counts collected, executed, passed, failed, skipped, and not run |
| Reproduction | Exact commands, nodeids, environment, and random seeds |

Do not claim that an operator is broadly proven correct merely because a small number of cases pass. Cases that are only designed, implemented, collected, or skipped do not count as executed coverage.

- `PASS`: Every mandatory applicable dimension has trustworthy executed evidence on the specified target, with no unresolved required contract gap.
- `FAIL`: Any mandatory correctness, invalid-input rejection, state, gradient, compilation/runtime, or other delivery requirement is violated.
- `NOT_VERIFIED`: Required evidence is missing, tests were not collected or run, tests were skipped, the environment blocked execution, or the contract remains unclear.

## Architecture Checklist

- **One entry point**: `tilelang-op-test`.
- **Three business scenarios**: delivery from scratch, existing-suite improvement, delivery acceptance.
- **Two pytest organization forms**: standalone tests and vLLM source-embedded tests.
- **Four core references**: [Delivery Workflows](references/delivery-workflows.md), [Test Credibility](references/test-credibility.md), [Coverage Design](references/coverage-design.md), and [Repository Test Execution](references/repository-testing.md).
- **Three tools**: `discover_tests.py` for static discovery, `check_st_spec.py` for contract/case review, and `pytest_st_evidence.py` for recording collection/execution evidence.
- **Three pilots**: `engram_hash`, SwiGLU, and `bmp_to_patches`.
- **Ten dimensions**: the coverage and evidence checklist in Step 4.

[Case Specification](references/case-specification.md) provides the structured format, and [Pilot Evidence](references/pilot-evidence.md) records examples. Both support the four core references without adding a new workflow category.

[Generalization Evidence](references/generalization-evidence.md) records usage after the pilots. These applications found or fixed genuine coverage gaps without changing the architecture checklist.

## Pilot Routing

When handling `engram_hash`, SwiGLU, or `bmp_to_patches`, read [Pilot Evidence](references/pilot-evidence.md). These examples respectively establish the initial paths for exact integer output, optional-parameter constraints, and source-embedded boundary tests on Ascend. For other operators, use the general workflow rather than copying pilot-specific shapes or tolerances.
