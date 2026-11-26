# Running Tests in the Current Repository

Read this reference before collecting or running tests. Check the current repository whenever any of the facts below may have changed.

## Test Entry Points and Configuration

Work from the current repository root and prefer test files supplied by the caller. Framework tests are typically located under `testing/ascend/`, while operator examples and their tests are under `examples/ascend/`; tests for generated operators remain in their artifact directory. Before running tests, read the applicable pytest configuration and every relevant `conftest.py` to confirm collection rules, backend parameters, markers, plugins, and device binding. Do not assume fixed `testpaths`, test tiers, or fail-fast plugins.

```bash
TILELANG_DEFAULT_TARGET=<backend> python -m pytest <test_file> --collect-only -q
TILELANG_DEFAULT_TARGET=<backend> python -m pytest <test_file> -x
```

Embedded `test_*` functions in example files may not match the default file-collection rules; in that case, specify the file or nodeid explicitly. Backend parametrization may include multiple targets, so verify the nodeids actually collected and their JIT configuration. Setting an environment variable alone does not prove that every case uses the target backend.

Static discovery provides only function-level hints. Obtain parametrized nodeids in the same environment used for formal execution, and validate them serially before running in batches. For final acceptance, run the complete target list without `-x` and retain the result of every case. Empty collection is not a passing result.

Set concurrency according to the devices actually available and the confirmed worker-binding and isolation mechanisms. Run serially when there is no binding mechanism or only one physical device. When multi-device support is confirmed, use at most one worker per device. Reduce concurrency after an OOM; do not reduce the case set. Follow the workflow entry-point contract for device access.

Apply benchmark or resource-marker filters only when the current tests actually define them, and record every exclusion and its rationale. Do not treat exit codes, test tiers, or memory-profiling conventions from an older project as universal pytest behavior.

Place oracles provided by optional dependencies in separate tests. Calling `pytest.skip()` after core assertions execute marks the entire nodeid as skipped, so it cannot serve as passing evidence. Report the core-reference result separately from the unavailable optional comparison.

## Backend Evidence

When applicable, record all three of the following:

- The input device selected at runtime;
- The host dispatch branch, such as `is_ascend()` or an equivalent condition;
- The compiler/JIT target, including a target explicitly embedded in a decorator.

An environment variable cannot override an explicit JIT target. When static dispatch alone cannot prove the execution path, use kernel source or profiling evidence.

### Bisheng and Host C++ Headers

If Ascend compilation fails inside standard C++ headers, first record the host compiler and the headers actually selected before deciding whether the operator is at fault. Bisheng/CANN may select incompatible GCC headers. In that case, locate an installed GCC version compatible with Bisheng and use `export CPLUS_INCLUDE_PATH=...` to point to that version's C++ header paths.

After setting it, rerun the exact same nodeid. If it then passes, that establishes that the earlier run was blocked by the compilation environment. Retain both runs in the execution record. If no compatible GCC installation can be found, stop ST acceptance and report the compilation-environment problem.

## Result Classification

Classify the following results separately:

- Test collection or dependency errors;
- Compilation or lowering errors;
- Correctness failures;
- Invalid-input rejection differing from expectations;
- Device, OOM, or resource errors;
- Timeouts or interruptions;
- skipped/xfail;
- passed.

Retain the original first failure and the exact command. A passing result obtained later with a different environment or random seed is additional evidence; it does not erase the earlier result.
