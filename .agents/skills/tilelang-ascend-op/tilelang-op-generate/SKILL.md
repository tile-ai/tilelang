---
name: tilelang-op-generate
description: Generate the TileLang operator and pytest for Ascend 950 (A5) based on user needs and design.md, and debug until the specified case accuracy passes. When there are problems in the design, the source code is corrected and implemented and the design is synchronized. The generation phase only delivers two Python files, kernel and test, and does not undertake generalized use case expansion and performance tuning.
---

# TileLang A5 operator generation

## 1. Goals and Scope

Implement user-specified cases to deliver TileLang kernel and accuracy tests that can run on the target NPU. The actual accuracy of all specified cases is passed as the completion condition, and there is no automatic expansion of the input range, test grading or performance tuning.

## 2. Input and basis

When entering the generation phase, first load Skills named `npu-arch`, `tilelang-op-design`, and `tilelang-performance-best-practices`. According to the target version and platform constraints in Section 3.1 of the design Skill, reuse the `full_soc`, `npu_arch` and complete `evidence` passed by the caller; when the evidence is missing, incomplete, the equipment or configuration changes, or the design only refers to typical specifications, `npu-arch` runs its own detection script to re-obtain it. Only the consistent evidence of Ascend950PR/DT series and `npu_arch=3510` is accepted, and then the available AIC/AIV core number and related buffer capacity are obtained through the runtime interface. Stop and report when any required Skill is not installed or cannot be loaded by name, without reading it through adjacent directory paths. Accessing the NPU follows the conventions in Section 4.3; query failures must not impersonate the physical machine value with static mapping or typical values.

Read user requirements, corresponding `design.md`, reference interfaces and specified tests, and clarify calculation semantics, input and output, cases, accuracy requirements and target backends. Existing information is used directly; when necessary semantics are missing or there is a conflict of requirements, the information is paused and attached with parameter descriptions and filling examples to guide users to make additions. Implementation parameters such as tiles, threads, and number of cores are determined by the model.

When there is no applicable design report, the loaded `tilelang-op-design` Skill is first used to generate the report. Existing reports will be implemented after self-checking according to the Skill's template and no additional approval is required.

**`design.md` is an important reference, but not the only basis. ** The design provides an initial implementation plan; when the actual generation finds that the API, resource budget, algorithm or synchronization plan is not established, the implementation will be adjusted based on the source code and compilation and accuracy verification results, and the affected design content will be revised simultaneously.

Basis for selection by question:

| Question | Basis |
|---|---|
| Computational semantics, interfaces and precision | User requirements and consistent formulas, reference implementations and specified tests; clarification in case of conflicts |
| Initial algorithm and implementation structure | Schemes, parameters and sources in `design.md` |
| API and backend feasibility | Actual version of TileLang API, lowering/codegen, related tests and current round of running results |
| Implementation Reference | The target operator, matching examples in `examples/ascend/`, and the matching operator-family `_asc.py` references routed by `tilelang-performance-best-practices` |

The source code, tests and examples are all based on the current warehouse root directory, following the root directory determined by the user or design, and do not rely on the warehouse name or personal absolute path. For frameworks, refer to `src/ascend/`, `src/backend/` and `tilelang/`, and for operator examples, refer to `examples/ascend/`. Use the loaded best-practices Skill's `references/index.md` to classify the operator, then read only the matching family guides and `_asc.py` implementation references. Those files contain Ascend host and kernel structures for source reading and do not require template-maturity registration. A reference that retains `target="pto"` remains an eligible Ascend implementation reference; do not use CUDA implementations. Check the actual imported TileLang path and version to avoid inconsistencies between the source code and the running environment. The backend runs according to the determined configuration and does not avoid problems by silently switching backends.

## 3. Two delivery files

By default, it is in the same directory as `design.md`. Existing file names or user-specified paths take precedence:

| Documentation | Content |
|---|---|
| `<op_name>.py` | TileLang kernel, necessary imports, JIT builder, compilation parameters and kernel auxiliary definitions |
| `test_<op_name>.py` | Specify case, input generation, output and workspace allocation, kernel call, independent Golden, precision assertion |

The kernel file does not contain golden, tests, benchmarks, device probes or command line entries. The definitions of multi-kernel solutions are placed in the same file; the host calling sequence is placed in an independent calling function of the test file, which can be reused for subsequent profiling.

Backend selection belongs to the running configuration and not to the operator implementation. Backends must not be pinned via `target="..."`, `target='...'`, or other equivalents in either deliverable; the `target` parameter must be omitted from the TileLang JIT decorator, builder, and compile API, and Python code must not set `TILELANG_DEFAULT_TARGET`. Select the backend through command environment variables during runtime: PTO uses `TILELANG_DEFAULT_TARGET=pto`, AscendC uses `TILELANG_DEFAULT_TARGET=ascend`. Switching the backend does not modify the operator or test source code.

test uses pytest to import the actual kernel in the same directory to avoid mistesting the old implementation with the same name in the warehouse. Tests are organized into specified cases by parameters and do not rely on test classification or automatic expansion of shapes. Allows the use of installed dependencies and necessary existing public tools, without adding new launchers, registration files, `conftest.py` or independent test reports. Temporary logs and compilation caches are not included as deliverables and are not included in the design body.

## 4. Generation and debugging process

### 4.1 Check the plan and implement it

First check the queried hardware parameters with the design, and constrain the execution mode, number of cores, tiles, pipeline and buffer allocation accordingly; if they are inconsistent, correct the implementation and synchronize the design according to Section 4.4. Hardware querying is completed during the development process, and device detection logic is not added to the delivery kernel.

Check API, dtype/layout, execution domain and handling restrictions along key operations, and prioritize reusing existing backend implementations. Use the matching bundled `_asc.py` references for host dispatch, tiling, buffer organization, forward/backward decomposition, and kernel structure; do not require those source-reading files to appear in `template_status.md`. The core calculation must be completed by this TileLang kernel, and PyTorch, ready-made operators or pre-computed results cannot be called to replace the functions to be implemented.

Implement task indexing, buffer allocation, initialization, synchronization and writeback. When specifying a case with a tail block, the valid range and reduction fill value are processed; when multiple tasks write the same result, the merge method is clear. After parameter adjustment, resource occupation and task coverage are recalculated, and the entire algorithm is not copied for each case.

### 4.2 Establish independent accuracy test

- Build Golden from formulas or trusted references, explicitly calculating dtypes, output transformations and attribute semantics.
- Use a reproducible data generation method that covers all specified input and attribute combinations; output, index, and in-place update results are all checked according to interface requirements.
- Comparison standards follow user or trusted testing conventions; use evidence-based design recommendations when no established standards exist. Distinguish between element-by-element numerical comparison and byte-by-byte comparison, and do not require floating point results to be bitwise identical by default.
- Golden and kernel use equivalent input; when there is an in-place update, prepare input copies separately to avoid mutual contamination.

Golden should not be reversely modified based on kernel output, relax thresholds, remove failing cases, or add skip/xfail to pass. If the reference test itself is wrong, correct it based on the calculation definition; if there is any ambiguity in requirements, clarify it first.

### 4.3 Compile, run and repair

First check that the file can be imported and the test can be collected, and then run the specified test on the target backend. If necessary, locate a single failed case first, and then return to all specified cases after repair; the final result must come from the latest kernel and test delivered.

The final accuracy verification uses the target backend passed in when calling or obtained from `design.md`, and only runs the corresponding command; after the fix, the same backend is still used to rerun all specified cases:

```bash
TILELANG_DEFAULT_TARGET=pto pytest test_<op_name>.py
TILELANG_DEFAULT_TARGET=ascend pytest test_<op_name>.py
```

Distinguish between environment, API/compilation, indexing, numerical and synchronization issues. Combine the error report, failure location, source code and necessary generated code to locate the cause, and then verify after completing the grounded repair. Technical issues are investigated by the model and are not passed to the user to select tiles or troubleshoot the backend.

Follow the equipment operating conventions for the current portal. In the Codex, commands to access the NPU are prefixed with `env -u ASCEND_RT_VISIBLE_DEVICES` and set with `sandbox_permissions=require_escalated`; the same applies when an import or test collection may initialize the NPU. Use a single worker when confirming only one physical device.

Continuously fix manageable issues. When equipment, dependencies or necessary user information are missing and the process cannot continue, the actual blocking and recovery conditions are clarified, static checks or skipped use cases are not used instead of precision passing, and no tuning is entered.

### 4.4 Synchronous design

If the implementation changes the algorithm, chunking, data flow, resource budget, key API or applicable conditions, update the relevant conclusions and basis of the original `design.md`; there is no need to rewrite the plan when only repairing the code to comply with the original design. Maintain user semantics, specified scope and precision standards.

The report retains final implementation guidance and reasons for necessary design changes. After revision, self-check according to the design skills and template, and the report directory will not be rewritten due to adjustment and implementation. Synchronous design does not increase the deliverable files during the generation phase.

## 5. Complete the check

Confirm when completed: the responsibilities of the two files are clear; the `target` of the delivered code is not fixed; all the specified cases are actually passed; the test hits the delivered kernel; the design is consistent with the final implementation.
