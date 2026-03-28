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
