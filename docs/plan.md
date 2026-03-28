# Fix CI Test Failures on `pipeline_refactor_0329` Branch

## Goal Description

Fix all compilation errors and correctness failures in the TileLang local CI test suite caused by the compiler pipeline refactoring on the `pipeline_refactor_0329` branch. The refactoring moved three passes — `PipelinePlanning`, `InjectSoftwarePipeline`, and `ProducerConsumerWarpSpecializedTiled` — to run before `LayoutInference` and `LowerTileOp` in `tilelang/engine/phase.py`. This changes when several C++ transform passes see the IR: they now operate on tile-op-level IR rather than lowered IR. The branch also renames/replaces passes (e.g., `WarpSpecialized` to `ProducerConsumerWarpSpecialized`, `InjectPTXAsyncCopy` to `LowerPTXAsyncCopy`, `InjectTmaBarrier` removed).

All fixes must be made within existing pass implementations. No new passes may be added and no pass reordering is permitted. A known fix (the `subtree_modified_` guard in `inject_pipeline.cc`) must be preserved.

The acceptance target is the local CI script: `./maint/scripts/run_local_ci_test.sh`, which runs pytest on `examples/` and `testing/python/` with parallel workers on NVIDIA H800 (sm_90a / Hopper) hardware.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Full CI pass — `./maint/scripts/run_local_ci_test.sh` completes with 0 failures and 0 errors (xfail/skip are acceptable)
  - Positive Tests (expected to PASS):
    - Run `./maint/scripts/run_local_ci_test.sh 2>&1 | tee run.log` — exit code 0, pytest reports 0 failed, 0 errors
    - All `examples/` tests pass (compilation + correctness)
    - All `testing/python/` tests pass (unit + integration)
  - Negative Tests (expected to FAIL):
    - A deliberately broken transform pass (e.g., removing the `subtree_modified_` guard) causes at least one test failure
    - Uncommitted debug artifacts (print statements) produce noisy output distinguishable from clean runs

- AC-2: No test modifications — no test expectations changed, no tests skipped, no xfail markers added
  - Positive Tests (expected to PASS):
    - `git diff main -- testing/ examples/ | grep -E '(xfail|skip|expected)'` returns empty (no test expectation changes)
    - All test files are unmodified relative to main branch (except for reverting debug artifacts)
  - Negative Tests (expected to FAIL):
    - Adding `@pytest.mark.skip` to any test file is detected as a violation

- AC-3: Preserved fix — the `inject_pipeline.cc` `subtree_modified_` guard remains intact
  - Positive Tests (expected to PASS):
    - `grep -n "subtree_modified_" src/transform/inject_pipeline.cc` returns the guard lines
    - The guard prevents `local.var` buffer promotion to kernel parameter (verified by the tests that originally exposed this bug)
  - Negative Tests (expected to FAIL):
    - Removing the `subtree_modified_` guard causes `local.var` buffer misplacement failures

- AC-4: No regressions — every test that passes on the main branch reference build also passes on `pipeline_refactor_0329`
  - Positive Tests (expected to PASS):
    - For any test failing on the branch, running the same test from `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` (reference build directory) confirms it passes on main
    - After all fixes, no test that passes on the reference build fails on the branch
  - Negative Tests (expected to FAIL):
    - Introducing a new regression (e.g., breaking a previously passing kernel) is detected by comparing against the reference baseline

- AC-5: Debug artifacts removed — all debugging changes cleaned up
  - AC-5.1: Debug print statements removed from `tilelang/engine/phase.py` (lines 192-193 and 195-196: `print("After PipelinePlanning")`, `print(mod)`, `print("After InjectSoftwarePipeline")`, `print(mod)`)
    - Positive: `grep -n "^    print" tilelang/engine/phase.py` returns no matches in `LowerAndLegalize`
    - Negative: Leaving any `print(mod)` call produces massive IR dumps during CI
  - AC-5.2: Debugging changes reverted in `examples/cast/example_group_per_split_token_cast_to_fp8.py` (`tilelang.disable_cache()` call removed)
    - Positive: `git diff main -- examples/cast/example_group_per_split_token_cast_to_fp8.py` shows no changes (file matches main)
    - Negative: `tilelang.disable_cache()` causes unnecessary recompilation, slowing CI

- AC-6: Architecture preserved — no new compiler passes added, no pass reordering in `phase.py`
  - Positive Tests (expected to PASS):
    - `git diff main -- tilelang/engine/phase.py` shows only removal of debug prints, no pass additions or reorderings beyond the branch's intentional restructuring
    - No new `.cc` files in `src/transform/`
  - Negative Tests (expected to FAIL):
    - Adding a new `tilelang.transform.NewPass()` call is detected as a violation
    - Reordering existing pass calls in `LowerAndLegalize` or `OptimizeForTarget` is detected as a violation

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. This task has a narrow, deterministic design: the pass ordering is fixed and the acceptance target is binary (CI green or not).

### Upper Bound (Maximum Acceptable Scope)

All failing tests are fixed with targeted, well-understood changes in existing C++ and Python pass implementations. Each fix is verified against reference build IR comparison for the affected kernel category. Debug artifacts are fully cleaned up. Fixes cover all kernel categories encountered in the test suite: TMA warp-specialized pipelines, cp.async software pipelines, non-pipelined Hopper kernels, and SIMT global-to-shared kernels. Each fix cluster is validated with focused tests before the final full CI run.

### Lower Bound (Minimum Acceptable Scope)

All CI tests pass with 0 failures and 0 errors. Debug artifacts are reverted. The `inject_pipeline.cc` `subtree_modified_` guard is preserved. Fixes may be minimal and targeted without exhaustive IR comparison for every kernel, as long as the test suite is green.

### Allowed Choices

- Can use: targeted edits to existing `.cc`/`.h`/`.py` transform files in `src/transform/` and `tilelang/`; reference build at `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` for IR/CUDA comparison; `cmake --build build -j$(nproc)` for C++ rebuilds; `tilelang.transform.get_pass_context()` and `tl.dump_ir` for debugging
- Cannot use: new transform passes; pass reordering in `phase.py`; test expectation changes; `@pytest.mark.skip` or `@pytest.mark.xfail` additions; test file modifications (except reverting debug artifacts)
- Debug artifact cleanup (removing prints from `phase.py`, reverting example file) is explicitly permitted as non-functional cleanup under AC-5, distinct from the "no pass changes" constraint in AC-6

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The root cause pattern for most failures is that compiler passes now see tile-op-level IR (before `LowerTileOp`) instead of lowered IR. The debugging workflow for each failure:

1. Run the failing test in isolation to get the error message
2. Compare IR/CUDA output with the reference build:
   - From the reference directory: `cd /weka-hg/prod/deepseek/permanent/wanglei/tilelang_old && python <test_script>`
   - From the branch: `cd /weka-hg/prod/deepseek/permanent/wanglei/tilelang && python <test_script>`
   - Diff the generated CUDA code or intermediate IR
3. Trace the pass pipeline using `tl.dump_ir` config to dump IR after each pass and locate the first divergence point
4. Fix the pass that produces incorrect output on tile-op IR — common patterns:
   - A pass assumes buffer access patterns only present in lowered IR
   - A pass does not recognize pipeline metadata in its new form (`tl.pipeline_context_num_stages` AttrStmt vs. `num_stages` For annotation)
   - A visitor does not handle tile-op nodes (e.g., `tl.tileop.copy`, `tl.tileop.gemm`)
5. Rebuild C++ (`cmake --build build -j$(nproc)`), rerun the specific test, then move to the next failure

### Relevant References

- `tilelang/engine/phase.py` — compiler pass pipeline definition; `LowerAndLegalize` (line 138) and `OptimizeForTarget` (line 228) are the two main pass sequences
- `src/transform/inject_pipeline.cc` — software pipeline injection; known fix at `subtree_modified_` guard (line ~1156, ~1200)
- `src/transform/pipeline_planning.cc` — pipeline planning pass (68KB); preserves `tl_pipelined_num_stages` annotation
- `src/transform/layout_inference.cc` — layout inference; handles both `For.annotations["num_stages"]` (line 653) and `AttrStmt` with `kPipelineContextNumStages` (line 819)
- `src/transform/lower_tile_op.cc` — tile operation lowering; consumes `tl.pipeline_context_num_stages` via stack (line 1123)
- `src/transform/producer_consumer_ws_tiled.cc` — tiled warp specialization; invokes `MultiVersionBuffer` internally (line 840)
- `src/transform/multi_version_buffer_rewriter.cc` — buffer versioning for pipelines
- `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` — reference main branch build for comparison
- `maint/scripts/run_local_ci_test.sh` — CI test runner (pytest-xdist parallel execution)

## Dependencies and Sequence

### Milestones

1. **Baseline Assessment**: Establish the current failure landscape
   - Phase A: Run full CI on `pipeline_refactor_0329` branch, collect all failures
   - Phase B: For any failing test, run it from the reference build directory to confirm it passes on main
   - Phase C: Categorize failures by root cause pattern (compilation error, correctness failure, crash) and by affected pass

2. **Debug Artifact Cleanup**: Remove non-functional debugging changes
   - Step 1: Remove `print("After PipelinePlanning")`, `print(mod)`, `print("After InjectSoftwarePipeline")`, `print(mod)` from `tilelang/engine/phase.py`
   - Step 2: Revert `examples/cast/example_group_per_split_token_cast_to_fp8.py` to its committed state (remove `tilelang.disable_cache()`)

3. **Systematic Fix Application**: Fix each failure cluster with targeted changes
   - Phase A: Fix compilation errors in transform passes (blocking — these prevent other tests from running). Rebuild C++ after each fix.
   - Phase B: Fix correctness failures in transform passes. Rebuild C++ after each fix.
   - Phase C: After each fix cluster, run focused tests for the affected kernel category to verify before proceeding

4. **Final Validation**: Confirm all acceptance criteria
   - Step 1: Run `./maint/scripts/run_local_ci_test.sh` — must complete with 0 failures, 0 errors
   - Step 2: Verify `inject_pipeline.cc` `subtree_modified_` guard is intact
   - Step 3: Verify no test files modified (except debug artifact reverts)

Milestone 2 has no dependencies and can run in parallel with Milestone 1. Milestone 3 depends on Milestone 1 (need failure categorization). Milestone 4 depends on Milestones 2 and 3.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Run `./maint/scripts/run_local_ci_test.sh` on `pipeline_refactor_0329`, collect full failure list with error messages | AC-1 | coding | - |
| task1b | For each failing test, run it from `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` to confirm it passes on main (establishes reference baseline) | AC-4 | coding | task1 |
| task2 | Categorize failures by root cause pattern: which pass produces incorrect output, what IR shape triggers the bug, which kernel category is affected | AC-1 | analyze | task1, task1b |
| task3 | Remove debug print statements from `tilelang/engine/phase.py` (lines 192-193, 195-196) | AC-5 | coding | - |
| task4 | Revert `examples/cast/example_group_per_split_token_cast_to_fp8.py` to committed state (remove `tilelang.disable_cache()`) | AC-5 | coding | - |
| task5 | Fix compilation errors in C++ transform passes; rebuild with `cmake --build build -j$(nproc)` after each fix | AC-1, AC-6 | coding | task2 |
| task5v | Run focused tests for each compilation-fix cluster (e.g., `pytest testing/python/transform/` for pipeline/WS fixes, `pytest examples/warp_specialize/` for WS kernel fixes) | AC-1 | coding | task5 |
| task6 | Fix correctness failures in C++ transform passes; rebuild with `cmake --build build -j$(nproc)` after each fix | AC-1, AC-4, AC-6 | coding | task2 |
| task6v | Run focused tests for each correctness-fix cluster (e.g., `pytest testing/python/kernel/` for kernel fixes, specific failing examples) | AC-1 | coding | task6 |
| task7 | Verify `inject_pipeline.cc` `subtree_modified_` guard is intact: `grep -n "subtree_modified_" src/transform/inject_pipeline.cc` | AC-3 | analyze | task5, task6 |
| task8 | Run full CI validation: `./maint/scripts/run_local_ci_test.sh` — must report 0 failures, 0 errors | AC-1, AC-2, AC-4 | coding | task3, task4, task5v, task6v |

## Claude-Codex Deliberation

### Agreements

- The `inject_pipeline.cc` `subtree_modified_` guard is a correct and necessary fix that must be preserved. It prevents outer block read/write region recomputation from pulling inner-block `local.var` buffer references into scope, which would mislead LCA analysis in `PlanAndUpdateBufferAllocationLocation`.
- The debug print statements in `phase.py` (lines 192-196) and the `tilelang.disable_cache()` in the example file are debugging artifacts that must be removed before CI validation.
- The architectural constraint of "no new passes, no pass reordering" is coherent with the branch goal and should be maintained. Debug artifact cleanup is explicitly separate from this constraint.
- The `producer_consumer_ws_tiled.cc` header comments (lines 16-19) about prerequisites are stale — the pass now invokes `MultiVersionBuffer` internally (line 840) — but this is a documentation issue, not a functional bug.
- `LayoutInference` already handles both pipeline metadata forms: `For.annotations["num_stages"]` (line 653) and `AttrStmt` with `kPipelineContextNumStages` (line 819). The metadata propagation gap initially flagged by Codex is not present.
- Performance regression is not a blocking criterion for this plan (user decision).
- Stale comment cleanup in touched files is out of scope (user decision).

### Resolved Disagreements

- **LayoutInference metadata handling**: Codex v1 claimed `LayoutInference` only tracks pipelined state from `For.annotations["num_stages"]` and the metadata propagation was incomplete. Claude provided code evidence that `LayoutInference` also handles `AttrStmt` with `kPipelineContextNumStages` (layout_inference.cc:819-841). Codex v2 accepted this correction. **Resolution**: No metadata gap exists; both code paths are functional.

- **Validation scope**: Codex v2 argued that AC-1 is too weak because `run_local_ci_test.sh` only covers local CUDA pytest, while full CI also includes `clang-tidy`, ROCm, Metal, and CuTeDSL jobs. Claude noted the draft explicitly specifies the local CI script as the acceptance target. **Resolution**: User confirmed local CUDA CI is the correct scope. Broader CI is out of scope.

- **AC-4 baseline method**: Codex v2 pointed out that overriding `PYTHONPATH` from the branch repo would not reliably exercise the reference native build. Claude revised the approach to run tests directly from the reference build directory. **Resolution**: Baseline established by running from `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` directory.

- **Scope contradiction**: Codex v2 noted the plan says "all fixes within existing pass implementations" but task3/task4 modify `phase.py` and an example file. Claude clarified that AC-6 applies to functional compiler passes, while AC-5 cleanup is explicitly whitelisted. **Resolution**: Constraint narrowed to "no new passes, no pass reordering"; debug cleanup is separate.

- **IR comparison overclaim**: Codex v2 noted the plan claimed "full IR comparison" without defining how it would be performed. Claude demoted IR comparison to a debugging methodology in Feasibility Hints, not an acceptance criterion. **Resolution**: IR comparison is a tool, not a coverage guarantee.

### Convergence Status

- Final Status: `converged`
- Rounds: 2
- Round 1: 5 REQUIRED_CHANGES identified, all addressed in v2
- Round 2: 1 REQUIRED_CHANGE (AC-4 baseline execution method), addressed by running from reference directory
- No remaining high-impact DISAGREE items

## Pending User Decisions

- DEC-1: CI scope — local CUDA CI only vs. full GitHub Actions matrix
  - Claude Position: Local CUDA CI matches the draft specification
  - Codex Position: Full CI matrix would catch more issues (clang-tidy, ROCm, Metal, CuTeDSL)
  - Tradeoff Summary: Local CI is faster and matches draft; full CI requires more hardware and is broader than the stated goal
  - Decision Status: `Local CUDA CI only` (user decided)

- DEC-2: Performance regression as blocking criterion
  - Claude Position: Not blocking; focus on correctness first, assess performance separately
  - Codex Position: "Correct but slower" kernels could still be a problem for a compiler pipeline refactor
  - Tradeoff Summary: Adding performance criterion would significantly expand scope and require baseline benchmarks; correctness is the immediate priority
  - Decision Status: `Not blocking` (user decided)

- DEC-3: Stale comment cleanup in touched C++ files
  - Claude Position: Out of scope; focus strictly on test failures
  - Codex Position: Updating comments in touched files prevents future confusion
  - Tradeoff Summary: Comment updates are low-risk but expand scope beyond "fix CI failures"
  - Decision Status: `Skip comment cleanup` (user decided)

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- Fix descriptions in commit messages should reference the affected pass and kernel category, not plan task IDs

### Build Requirements

- After any `.cc` or `.h` file edit: `cmake --build build -j$(nproc)`
- After any `.py` file edit in `tilelang/`: changes take effect immediately (no rebuild needed)
- Before final CI run: ensure a clean rebuild has been done for all accumulated C++ changes

### Environment

- Hardware: 5x NVIDIA H800 (sm_90a / Hopper)
- Branch: `pipeline_refactor_0329`
- Reference build: `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` (main branch, pre-built)
- CI script: `./maint/scripts/run_local_ci_test.sh` (pytest-xdist, auto-detects GPUs, default 4 workers per device)

--- Original Design Draft Start ---

# Draft: Fix CI Test Failures on `pipeline_refactor_0329` Branch

## Background

The `pipeline_refactor_0329` branch refactors the TileLang compiler pipeline by moving `PipelinePlanning` + `InjectSoftwarePipeline` + `ProducerConsumerWarpSpecializedTiled` before `LayoutInference` and `LowerTileOp`. This is a major restructuring that changes the order of compiler passes in `tilelang/engine/phase.py` and replaces several C++ transform passes (e.g., `WarpSpecialized` -> `ProducerConsumerWarpSpecialized`, `InjectTmaBarrier` removed, `InjectPTXAsyncCopy` -> `LowerPTXAsyncCopy`).

While the refactor is architecturally sound, it has introduced regressions that cause some kernels to fail compilation or produce incorrect results.

## Goal

Make the full local CI test suite pass:

```bash
./maint/scripts/run_local_ci_test.sh 2>&1 | tee run.log
```

This script runs `pytest` on two directories:
1. `examples/` -- end-to-end example tests (compilation + correctness)
2. `testing/python/` -- unit and integration tests for language features, transforms, codegen, etc.

## Environment

- **Hardware**: 5x NVIDIA H800 (sm_90a / Hopper)
- **Branch**: `pipeline_refactor_0329`
- **Reference**: A working copy at `/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` (main branch build) can be used to compare generated IR or CUDA code for any failing kernel.
- **Build**: C++ changes are in `build/` -- rebuild with `cmake --build build -j$(nproc)` after any `.cc`/`.h` edits.

## Known Issue Already Fixed

- **`local.var` buffer promoted to kernel parameter**: `InjectSoftwarePipeline` unconditionally recalculated block reads/writes annotations, embedding inner-block `local.var` buffer references into outer blocks. This misled the LCA analysis in `PlanAndUpdateBufferAllocationLocation`, causing `local.var` buffers to become kernel parameters instead of local variables. Fixed by guarding reads/writes recalculation with a `subtree_modified_` flag in `inject_pipeline.cc`.

## Debugging Strategy

For any failing test:

1. **Identify the failing kernel** from pytest output.
2. **Compare IR/CUDA** with the reference build:
   - Set `PYTHONPATH=/weka-hg/prod/deepseek/permanent/wanglei/tilelang_old` and run the same example to get the working output.
   - Set `PYTHONPATH=/weka-hg/prod/deepseek/permanent/wanglei/tilelang` for the broken output.
   - Diff the generated CUDA code or intermediate IR to locate the divergence.
3. **Trace the pass pipeline**: Use `tilelang.transform.get_pass_context()` with `tl.dump_ir` config or insert print statements in `phase.py` to dump IR after each pass.
4. **Root cause**: Typically the issue is a pass that doesn't handle tile-op-level IR correctly (since passes now run earlier, before `LowerTileOp`), or a pass ordering issue.
5. **Fix in C++** (`src/transform/*.cc`) or **Python** (`tilelang/engine/phase.py`, `tilelang/transform/__init__.py`).
6. **Rebuild and re-test** the specific failing test before moving on.

## Scope & Constraints

- Fix all compilation errors and correctness failures in the CI suite.
- Do NOT change test expectations or skip tests -- the goal is to make the existing tests pass.
- **Do NOT add new passes or adjust pass ordering in `phase.py`** -- the current pass sequence is intentional. All fixes must be made within the existing pass implementations (i.e., edit the current `.cc`/`.h`/`.py` files, not introduce new transform passes or reorder them).
- Minimize changes to the pipeline architecture; prefer targeted fixes in individual passes.
- The `examples/cast/example_group_per_split_token_cast_to_fp8.py` has a `tilelang.disable_cache()` + `exit()` added during debugging -- revert this file to its committed state before running CI.

## Acceptance Criteria

- `./maint/scripts/run_local_ci_test.sh` completes with 0 failures and 0 errors (xfail/skip are acceptable).
- No regressions: tests that passed on main should still pass.
- The `inject_pipeline.cc` fix (subtree_modified_ guard) must be preserved.

--- Original Design Draft End ---
