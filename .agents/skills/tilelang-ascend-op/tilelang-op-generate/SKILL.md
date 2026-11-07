---
name: tilelang-op-generate
description: Generate a one-kernel-per-public-entry TileLang operator and pytest for Ascend 950 (A5) based on user needs and design.md, and debug until the specified case accuracy passes. When there are problems in the design, the source code is corrected and implemented and the design is synchronized. The generation phase only delivers two Python files, kernel and test, and does not undertake generalized use case expansion and performance tuning.
---

# TileLang A5 operator generation

## 1. Goals and Scope

Implement user-specified cases so that each independent computation direction or public computation entry has exactly one logical TileLang kernel and its accuracy tests on the target NPU. Forward and backward are separate entries and may own different PrimFuncs; one invocation of either entry must launch its kernel exactly once. The actual accuracy of all specified cases is passed as the completion condition, and there is no automatic expansion of the input range, test grading or performance tuning.

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
| `<op_name>.py` | For each declared public computation entry, one target factory with exactly one nested TileLang `@T.prim_func`, plus necessary imports, compilation parameters and in-kernel auxiliary definitions |
| `test_<op_name>.py` | Specify case, input generation, output and workspace allocation, kernel call, independent Golden, precision assertion |

The kernel file does not contain golden, tests, benchmarks, device probes or command line entries. The counting unit is each public computation entry, not the entire file: forward and backward may each define their own factory and PrimFunc. Within one entry, the file must not define or compile a candidate or shape-specific second kernel. Python factory parameters may select compile-time configurations, and `T.macro` may factor reusable in-kernel code; after expansion, that entry must still have one `@T.prim_func`. Different shapes may select a small number of JIT parameter combinations from that same kernel definition; the test or public wrapper must not dispatch the entry among multiple factories or PrimFuncs or chain multiple kernel launches.

Backend selection belongs to the running configuration and not to the operator implementation. Backends must not be pinned via `target="..."`, `target='...'`, or other equivalents in either deliverable; the `target` parameter must be omitted from the TileLang JIT decorator, builder, and compile API, and Python code must not set `TILELANG_DEFAULT_TARGET`. Select the backend through command environment variables during runtime: PTO uses `TILELANG_DEFAULT_TARGET=pto`, AscendC uses `TILELANG_DEFAULT_TARGET=ascend`. Switching the backend does not modify the operator or test source code.

test uses pytest to import the actual kernel in the same directory to avoid mistesting the old implementation with the same name in the warehouse. Tests are organized into specified cases by parameters and do not rely on test classification or automatic expansion of shapes. Allows the use of installed dependencies and necessary existing public tools, without adding new launchers, registration files, `conftest.py` or independent test reports. Temporary logs and compilation caches are not included as deliverables and are not included in the design body.

## 4. Generation and debugging process

### 4.1 Check the plan and implement it

First check the queried hardware parameters with the design, and constrain the execution mode, number of cores, tiles, pipeline and buffer allocation accordingly; if they are inconsistent, correct the implementation and synchronize the design according to Section 4.4. Hardware querying is completed during the development process, and device detection logic is not added to the delivery kernel.

Audit the design and existing source before implementation, after any edit that changes factories, wrappers, or dispatch, and again before completion. Enumerate every independent computation direction/public entry and record its public symbol, target factory, nested PrimFunc, specified cases, and invocation site. Use Python AST inspection where the structure is statically represented, supplemented by source call-graph inspection for decorators or indirection that AST alone cannot resolve. The audit must prove for every entry that:

- its target factory contains exactly one nested `@T.prim_func`;
- all of its specified cases resolve to that factory and PrimFunc, although the factory may specialize general parameters such as tile sizes or core counts;
- its public call path invokes the compiled kernel exactly once and contains no candidate-PrimFunc selection, host shape dispatch among multiple kernels, or chained launch;
- every delivered operator PrimFunc belongs to a declared public entry, so an unused or fallback candidate cannot escape the per-entry count.

If any mapping or launch count remains ambiguous, stop and report the unresolved source path instead of assuming compliance. General execution paths may branch inside an entry's kernel on interface semantics or provable properties such as dtype, capacity, alignment, task count, full/tail tiles, contiguity, or hardware resource limits. Report the final entry/factory/PrimFunc/launch mapping and audit conclusion in the completion response without adding a third delivery file.

Check API, dtype/layout, execution domain and handling restrictions along key operations, and prioritize reusing existing backend implementations. Use the matching bundled `_asc.py` references for host-side argument/configuration handling, tiling, buffer organization, forward/backward structure, and kernel structure, but adapt any multi-kernel reference to this Skill's per-entry one-kernel contract; do not require those source-reading files to appear in `template_status.md`. The core calculation must be completed by this TileLang kernel, and PyTorch, ready-made operators or pre-computed results cannot be called to replace the functions to be implemented.

Implement task indexing, buffer allocation, initialization, synchronization and writeback. When specifying a case with a tail block, the valid range and reduction fill value are processed; when multiple tasks write the same result, the merge method is clear. After parameter adjustment, resource occupation and task coverage are recalculated, and the entire algorithm is not copied for each case.

Reject any implementation attempt that needs a second device kernel for the same public entry, even if it compiles or passes accuracy. When revising an existing entry, restore or retain its latest correctness-passing single-kernel implementation without rolling back valid kernels for other directions. When generating an entry from scratch and its required semantics cannot be implemented with one kernel using the current TileLang API and selected-backend lowering, report the exact API, synchronization, capacity, or lowering blocker and stop; do not use a same-entry multi-kernel or host-compute fallback, and do not remove specified cases.

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

Confirm when completed: the responsibilities of the two files are clear; the `target` of the delivered code is not fixed; every declared public computation entry/direction maps to exactly one target factory and one nested `@T.prim_func`; each of its specified cases uses that same logical kernel and performs exactly one runtime launch; no same-entry candidate PrimFunc, host shape dispatch, chained kernel, or host-compute fallback exists; the final source/AST audit is unambiguous and reported; all specified cases actually pass; the test hits the delivered kernels; and the design is consistent with the final implementation.
