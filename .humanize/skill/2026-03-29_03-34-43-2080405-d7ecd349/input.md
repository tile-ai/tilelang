# Ask Codex Input

## Question

You are reviewing CANDIDATE PLAN v2 (revised) for the TileLang project. This is Round 2 of convergence review. Check if the revisions address Round 1 feedback.

## Changes from v1 to v2

### Revisions addressing Round 1 REQUIRED_CHANGES:

1. VALIDATION SCOPE CLARIFIED: The draft explicitly specifies './maint/scripts/run_local_ci_test.sh' as the acceptance target. The plan now explicitly states scope is local CUDA pytest only. Full GitHub CI (ROCm, Metal, clang-tidy, CuTeDSL) is out of the plan's scope per the user's draft specification. Whether broader CI matters is deferred to user decision.

2. AC-4 REGRESSION METHOD ADDED: New task 'task1b' establishes a main-branch baseline by running the same CI script with PYTHONPATH pointing to the reference build at /weka-hg/prod/deepseek/permanent/wanglei/tilelang_old. The baseline failure set is compared against the branch failure set after fixes. AC-4 is satisfied when no test that passes on main fails on the branch.

3. SCOPE CONTRADICTION RESOLVED: The architectural constraint ('no new passes, no pass reordering') applies to functional compiler passes in the pipeline. AC-5 cleanup (removing debug prints from phase.py, reverting example file) is explicitly categorized as 'debug artifact cleanup' — not a functional pass change. This is now stated in the plan's Allowed Choices section.

4. INTERMEDIATE VERIFICATION ADDED: New milestone 2.5 'Focused Verification' requires running targeted test subsets after each fix cluster:
   - After TMA/warp-specialized fixes: run testing/python/transform/ tests and examples/warp_specialize/
   - After cp.async fixes: run relevant transform tests
   - After general pipeline fixes: run testing/python/kernel/ tests
   Only after all clusters pass does full CI run (task8).

5. IR COMPARISON ADJUSTED: Removed 'full IR comparison' claim. IR/CUDA comparison against reference build is a debugging methodology (described in Feasibility Hints), not a coverage guarantee or acceptance criterion.

6. REBUILD STEP ADDED: Explicit cmake rebuild step (cmake --build build -j$(nproc)) added as a prerequisite after any C++ file changes. This is now part of task5 and task6 descriptions.

### Updated Acceptance Criteria:
- AC-1: ./maint/scripts/run_local_ci_test.sh completes with 0 failures, 0 errors (xfail/skip OK)
- AC-2: No test expectations changed, no tests skipped
- AC-3: inject_pipeline.cc subtree_modified_ guard preserved
- AC-4: No test that passes on main (reference build) fails on pipeline_refactor_0329 — verified via baseline comparison
- AC-5: Debug artifacts removed: print statements in phase.py (lines 192-196), tilelang.disable_cache() in example file. Categorized as cleanup, not functional pass change.
- AC-6: No new compiler passes added, no pass reordering in phase.py

### Updated Task Breakdown:
| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Run full CI on branch, collect failure list | AC-1 | coding | - |
| task1b | Run CI with reference build to establish main baseline | AC-4 | coding | - |
| task2 | Categorize failures by root cause, compare with baseline | AC-1,AC-4 | analyze | task1,task1b |
| task3 | Remove debug prints from phase.py | AC-5 | coding | - |
| task4 | Revert example debugging changes | AC-5 | coding | - |
| task5 | Fix compilation errors (rebuild C++ after each fix) | AC-1,AC-6 | coding | task2 |
| task5v | Run focused tests for fix cluster (TMA/WS/cp.async) | AC-1 | coding | task5 |
| task6 | Fix correctness failures (rebuild C++ after each fix) | AC-1,AC-4,AC-6 | coding | task2 |
| task6v | Run focused tests for fix cluster (kernels/examples) | AC-1 | coding | task6 |
| task7 | Verify inject_pipeline.cc fix preserved | AC-3 | analyze | task5,task6 |
| task8 | Run full CI validation (final) | AC-1,AC-2,AC-4 | coding | task3,task4,task5v,task6v |

### Pending User Decisions (carried from Round 1):
- DEC-1: Is scope local CUDA CI only, or full GitHub Actions matrix?
- DEC-2: Is performance regression checking in scope?
- DEC-3: Should stale comments in touched files be updated?

## Your Review Task

Evaluate whether the v2 revisions adequately address Round 1 REQUIRED_CHANGES. Respond in EXACTLY this format:

AGREE: points accepted
DISAGREE: points still unreasonable
REQUIRED_CHANGES: remaining must-fix items (if any)
OPTIONAL_IMPROVEMENTS: non-blocking
UNRESOLVED: items needing user decision

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-03-29_03-34-43
