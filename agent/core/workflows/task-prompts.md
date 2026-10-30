# Subagent Invocation Parameter Details

This document is the **single execution manual** for all tilelang-tuning stages and Subagent invocations. The main agent executes profiling-entry generation in Step 0, as well as Steps 0.5, 2b, 2c, and 2.5. Steps 1 and 2a are dispatched by the subagent `name`, with the corresponding message template passed in full.

The entire workflow uses the `{backend}` and `{tilelang_target}` selected at entry: `pto → pto`, `ascendc → ascend`. Every compilation, pytest, and profiling command must explicitly set `TILELANG_DEFAULT_TARGET={tilelang_target}`. Defaulting, falling back, or mixing backends is prohibited.

---

## Step 0 Additional Step: Generate a Standalone Profiling Operator File (Required)

> Purpose: Generate a standalone profiling file that can run all user-provided cases in a single process.

### Preconditions

`{cases_csv}` is provided directly by the user and has passed the 1–20 case gate. `{operator_file}` is the `.py` file containing the target kernel definition in the current baseline. Generate it when Step 0 finishes in round 1; before each subsequent round, regenerate it from that round's latest baseline.

### Generation Rules

1. Copy the **complete original byte content unchanged** from `{operator_file}` as the prefix of `{output_dir}/round{N}/profiling_entry/baseline_<operator_name>_profiling.py`. Do not delete, reorder, format, or rewrite the kernel.
2. Append only a host-side profiling entry point after that prefix:
   - Write all 1–20 user-provided cases from `{cases_csv}` into `CASES` in their original order. Every case ID, parameter name, and value must match exactly.
   - Reuse the input-construction semantics from `{test_file}`, but ultimately call the copied local kernel in the generated file explicitly. Do not call a public wrapper that reroutes to the original project or to an old kernel from an installed package.
   - The host entry point accepts only `--device`, selects the device before creating any NPU tensor, and calls the target kernel exactly once per case in order within one process. A `--case-id` branch and per-case subprocesses are prohibited.
3. Step 0 only generates the file and performs static checks. Do not run pytest, a benchmark, or the kernel.
4. Statically verify that target/dispatch in `{operator_file}`, the wrapper, and `{test_file}` can enter `{tilelang_target}`. Stop if a hard-coded conflict is found; do not rewrite the source or switch backends.
5. Record the backend, `{operator_file}` SHA256, file byte length, `{cases_csv}` SHA256, generated-file path, and expected kernel name. Hard gate: the first `source_length` bytes of the generated file must be byte-for-byte identical to `{operator_file}`; embedded case IDs, parameter values, and order must exactly match `{cases_csv}`.

### Highest-Priority Stale-Artifact Gate

> ⚠️ **Never reuse an old profiling file to measure new source code.** Step 1 of every round must regenerate the file from that round's current baseline source. Step 2b must separately regenerate standalone profiling files from the latest target operator source of the round's baseline and each `optimized_<solution>` that passed the compilation and correctness gates. Do not copy, rename, patch, or continue using a profiling file from Step 0, a previous round, or another solution. If any source-prefix SHA, cases.csv SHA, source-relative path, or runtime kernel name does not match, stop collection immediately and regenerate from the corresponding latest source.

---

## Step 0.5: Pre-Optimization Correctness Baseline

> Purpose: Establish the **unoptimized baseline** correctness as the regression reference before any analysis or optimization action.

### Preconditions
Step 0 has validated the user-provided `{cases_csv}`. `{code_dir}` is the source-project root, and `{test_file}` is the unified pytest correctness test file in that project. The main agent executes this step directly.

```text
Establish the [pre-optimization correctness baseline] for the following operator:

- Baseline project directory: {code_dir}
- Correctness test file: {test_file}
- Output directory: {output_dir}
- Backend: {backend} (TILELANG_DEFAULT_TARGET={tilelang_target})

[Tasks]
1. Complete the NPU device-binding preflight before running pytest:
   - Run `npu-smi info -l`. Parse IDs that fully match `^\s*NPU ID\s*:\s*(\d+)\s*$` and their corresponding `Product Name` by device block. Exclude entries whose product name is empty or `NA`; do not infer the device count from the total number of nonempty output lines.
   - If the parent process already sets `ASCEND_RT_VISIBLE_DEVICES`, its valid list of nonnegative integer IDs is the upper bound of candidates; do not add devices outside that list. Otherwise, use valid entries from `npu-smi` as candidates. If candidates span product models, select the largest same-model group; break ties by choosing the group containing the smallest ID.
   - Selected IDs must be nonempty, unique, and nonnegative integers. Launch exactly **one** standalone preflight process with `ASCEND_RT_VISIBLE_DEVICES=<selected_ids>`, import `torch`/`torch_npu`, require `torch.npu.device_count()` to equal the selected count, and allocate a one-element NPU tensor followed by `torch.npu.synchronize()` on every logical device. Stop if any device fails; do not probe IDs one by one, and do not launch pytest.
   - Record in the report the candidate source, `npu-smi` IDs/product names, excluded entries, final selected IDs, batch runtime-verification results, and final worker count. Subsequent collection, pytest, and correctness regressions for every optimization solution of this operator must use the same command-level `ASCEND_RT_VISIBLE_DEVICES=<selected_ids>`.
2. Run `TILELANG_DEFAULT_TARGET={tilelang_target} python -m pytest {test_file} --collect-only -q` with the target backend and record the complete list of correctness nodeids. Exclude performance cases only if the current test actually defines a benchmark marker, and record the evidence for exclusion. Do not filter cases based on an unverified marker or test level.
3. Execute all correctness cases in the list. Final acceptance must not use `-x`. Run `TILELANG_DEFAULT_TARGET={tilelang_target} python -m pytest {test_file}`; add `-n <workers>` only after confirming xdist and the worker device-binding/isolation mechanism, with concurrency no greater than the verified available-device count. Otherwise, run serially. When performance cases must be excluded, explicitly pass the collected correctness nodeids. If OOM occurs, reduce concurrency without dropping cases. If collection is empty, stop and report it; do not treat it as a pass.
4. Use the actual assertions in `{test_file}`, such as `assert_close`, `assert_equal`, and `calc_diff`, as the correctness criteria. Do not modify or relax thresholds.
5. Compute the SHA256 of the baseline `{test_file}`. All later stages must use a byte-identical pytest file to keep pre- and post-optimization correctness criteria identical.
6. Produce `{output_dir}/precision-baseline.md`: use the common report format to record the backend, pytest file path and SHA256, device-binding preflight, complete collection list, actual pytest command, correctness criteria, and per-case PASS/FAIL, dtype, and shape.
7. Gate: the baseline passes only if the device preflight and collection-coverage validation pass and every listed correctness case passes. If any gate fails, set `Status: ❌ Failed` in the report and stop.

[Return]
- Path to `{output_dir}/precision-baseline.md`
- Whether the baseline is fully green (yes → proceed to Step 1; no → list nonconforming cases)
```

The main agent proceeds to Step 1 only after confirming that `precision-baseline.md` is fully green. If the baseline does not meet the criteria, stop and report to the user.

---

## Step 1: Performance Data Collection and Analysis

### Subagent Invocation Contract

The main agent uses the current host's native subagent dispatch capability, requests the subagent with logical name `tilelang-perf-analysis-expert`, and passes the following message in full. Replace placeholders only; do not rewrite the body.

```text
Collect and analyze performance data for the operator code to be tuned, executing these three stages in order: **run the operator and collect data → analyze performance → output tuning solutions**.

- Framework source root: {repo_root} (current repository; verify the actual imported version)
- Source-project root: {code_dir} (complete project root of the current round's baseline)
- Target operator file: {operator_file} (source file containing the target kernel definition in the current baseline)
- Standalone profiling operator file: {profiling_file} (generated by the main agent from the current round's latest `{operator_file}` and `{cases_csv}`; use only for current-round baseline collection)
- Test-case file: {cases_csv} (provided directly by the user and validated as 1–20 complete cases; do not derive, supplement, or filter it from pytest)
- Output directory: {output_dir} (operator-isolated directory; store all artifacts such as performance data and reports beneath it. Store current-round artifacts in the `{output_dir}/round{N}/` subdirectory, where N is the current round number)
- Backend: {backend} (TILELANG_DEFAULT_TARGET={tilelang_target}; every command run by the Skill named `tilelang-op-profiling` uses this value)
- Hardware-detection evidence: {hardware_evidence} (complete JSON for the same target device, including `full_soc`, `npu_arch`, and provenance; only Ascend950PR/DT + 3510 is accepted)

---

### Stage 1: Run the Operator and Collect Performance Data

First load the Skill named `npu-arch` to validate and reuse `{hardware_evidence}`. If the evidence is missing, incomplete, or the device/configuration has changed, use that Skill's bundled detection script to reacquire it. Then load the Skill named `tilelang-op-profiling` and pass it the complete evidence. If either Skill is not installed or cannot be loaded by name, stop and report.

1. **Verify that the profiling file matches the current round's latest source before compiling and running it**:
   - Verify that the source prefix of `{profiling_file}` is byte-for-byte identical to `{operator_file}`, and record both SHA256 values. Verify that embedded CASES match the IDs, parameters, and order in `{cases_csv}`.
   - If any validation fails, stop immediately and ask the main agent to regenerate the file from the current round's latest `{operator_file}`. Do not reuse, patch, or copy an old profiling file yourself.
   - Follow Step 1 of the `tilelang-op-profiling` Skill to run `{profiling_file}` directly, ensuring that all cases execute in a single process.
   - After compilation, **verify a one-to-one correspondence between kernel-mangled names captured by msprof and kernel function definitions in `{operator_file}`**, ensuring that collection targets the current profiling file's kernel rather than an old kernel from the original project, an installed package, or a cache.
2. Follow Step 2 of the `tilelang-op-profiling` Skill to launch msprof once, with `--launch-count` equal to the number of cases.
   - **Do not select only representative cases**: every case matters to the user and must be collected.
   - Do not restart Python or msprof per case. Numeric launch directories correspond to CASES in order.
   - If the `tilelang-op-profiling` Skill takes a fallback path, it must still run every CASE in this profiling file; do not substitute another probe.
   - Every test case must run on the NPU; host-side short-circuiting around the kernel is prohibited.
   - **Do not replace msprof collection with host-side timing (`std::chrono`, `gettimeofday`, and so on).**
   - **Do not collect baseline performance indirectly through a third-party evaluation framework such as `cann_bench eval`**. The `tilelang-op-profiling` Skill must collect baseline performance directly from the local kernel copied from the current round's latest `{operator_file}` into `{profiling_file}`, avoiding third-party host-side wrapper overhead as an unknown variable.
3. Obtain each case's aic/aiv elapsed times from profiling data and establish a consistent `kernel_time`: use aiv_time for AIV-only, aic_time for AIC-only, and the critical-path kernel time defined by the `tilelang-op-profiling` Skill for mixed AIC/AIV. Use the same timing basis for a case's baseline and all later solutions.

[Stage 1 Outputs]
- Performance-data artifacts for **all cases** (profiling directory), stored under `{output_dir}/round{N}/perf_per_case/`.
- A per-case kernel_time summary table that records the raw aic/aiv fields and timing basis.
- **Kernel-name validation record**: list correspondences between kernel-mangled names captured by msprof and kernel function definitions in the current round's latest `{operator_file}`.
- **Backend validation record**: record `{backend}`, `TILELANG_DEFAULT_TARGET={tilelang_target}`, and the actual compilation target.
- **Profiling-provenance validation record**: record `{operator_file}`, `{profiling_file}`, their source-prefix SHA256 values, `{cases_csv}` SHA256, and the consistency conclusion.
- Stop on collection failure; do not proceed to Stage 2.

---

### Stage 2: Performance Analysis

1. Load the Skill named `tilelang-perf-optimize`. Use that Skill for the PTO backend; for the AscendC backend, reuse only rules with direct evidence for `target=ascend`. Form strategy directions based on source code, lowering for the selected backend, and profiling.
   - Optimization points and templates in the Skill are candidate sources and examples, not a closed solution set. Use the current source, all cases, the resource model, and profiling evidence as the primary thread. You may transfer, combine, or extend similar principles, or propose candidates absent from the Skill. Even if a template matches, do not ignore non-template candidates directly related to measured bottlenecks. New candidates must still satisfy the falsifiable-hypothesis and evidence gates of the `tilelang-perf-optimize` Skill.
2. **Template matching**: load the Skill named `tilelang-performance-best-practices` and use only templates verified as compatible with the selected backend. If either Skill is not installed or cannot be loaded by name, stop and report.
   - **Determine the operator family before matching**: infer the operator family (for example, MatMul, Reduction, Elementwise, Broadcast, Conversion, or Scalar) from the current round's latest target operator source (compute structure and key APIs) and cases.csv (shape/dtype/attribute characteristics), then locate the corresponding family's guidance and templates in best practices.
   - **Classify the operator family by each case's compute pattern, not by locking the operator name to a single family**: different cases of one operator may span multiple families (for example, an axis with broadcasting → Broadcast family, purely elementwise → Elementwise family). Map each pattern to its corresponding family and search templates separately.
   - **Read only required references according to routing**: first read `references/index.md`, `references/template_status.md`, and the guide/decision tree for the matching operator family. Then read only the `.md` and `.py` files referenced by the branches matched by current cases. If cases span several compute patterns, route through each corresponding operator family separately. Do not recursively load unrelated families or every file not referenced by the current branch.
   - If a TileLang `.py` template is found, provide its complete path and read its maturity, validated scope, and performance evidence from `references/template_status.md`: templates marked `PRODUCTION_REFERENCE`, or `VERIFIED` for the current version with the optimized structure implemented, may be labeled "✅ Optimization pattern can be copied directly"; implementations marked `EXECUTABLE_BASELINE`, `PARTIAL`, or lacking evidence for the corresponding optimization/performance must be labeled "⚠️ Executable baseline or partial implementation"; material marked `DESIGN_ONLY`, or with no executable TileLang kernel for the current version, must be labeled "❌ Design reference only". Code in other programming models is not a template or API reference for this workflow.
   - If no template code is found, label it "No catalog reference". If the current repository or actual TileLang source provides locatable direct evidence for the API, lowering, correctness, and applicable scope, use that evidence to assess solution admissibility independently; do not automatically downgrade it to `DESIGN_ONLY` merely because the template-status table does not list it.
3. **API availability validation**: follow the source-of-truth priority in the `tilelang-performance-best-practices` Skill, together with invocation patterns that already compile and run in the current repository. Locate the framework source for the current repository, verify the actually imported version, and inspect relevant API definitions, compilation constraints, `examples/ascend/`, and lowering for the selected backend. Do not fill in APIs from memory. Parameters without direct evidence from API definitions, lowering, an in-repository runnable implementation, or target-version compilation results must not appear in an `IMPLEMENTABLE` solution; label them `DESIGN_ONLY` when only design evidence exists.
4. **Per-case analysis**: analyze the performance bottleneck of **every case** in `{cases_csv}`:
   - Give the case's bound type (VEC/MEM/SCALAR BOUND) and concrete bottleneck metrics.
   - Group cases with the same bottleneck characteristics and explain them collectively within the group.
   - **Every case must appear in the report**; none may be omitted or replaced merely with "same as above".
5. Case-grouping dimensions, in priority order: dtype → bound type → shape scale (S/M/L) → special-value characteristics.
6. Read the complete [Performance Optimization Search Coverage Gate](optimization-search-coverage/optimization_search_coverage_gate.md). Create `{output_dir}/round{N}/optimization-search-coverage.json` from the starter template, recording every case, pending adjudication obligation, and `CREATED` events for candidates admitted into solutions. Run ordinary validation to check record structure and reference consistency. Ordinary validation permits `PENDING`; do not block solution output or entry into Step 2 merely because an optimization direction has not yet been tested.

[Stage 2 Outputs]
- Multiple performance-tuning strategy directions and corresponding template-matching results, including per-case selection decisions for Type-B families.
- Bottleneck-analysis results for all cases, displayed by group while listing every case independently.
- Stop on analysis failure; do not proceed to Stage 3.

---

### Stage 3: Output the Performance Tuning Plan

> ⚠️ Use the Stage 2 analysis conclusions as input. **Do not** redo analysis or modeling.

1. **Output the Performance Tuning Plan report**:
   - Include at most three admitted performance-tuning solutions, whose status may only be `IMPLEMENTABLE` or `EXPERIMENT`, ordered by evidence strength and expected impact.
   - List `DESIGN_ONLY` items separately under "Unadmitted candidates", explaining missing TileLang API, selected-backend lowering, or verification evidence. Explain missing executable-template evidence only when a candidate depends on a bundled reference. These items do not count against the three-solution limit and are not passed to the implementation expert.
   - Each solution includes its optimization objective, tuned Tiling parameters, tuning strategy, and factual basis. When reusing a bundled reference, give the Skill name, internal reference-file relative path, **template skeleton**, and **template branch conditions**. When no template is reused, provide source paths in the current repository or actual TileLang source, direct API/lowering evidence, and the label "No catalog reference".
   - **Per-case coverage table**: list which cases the solution optimizes, each case's expected improvement direction, and its **template branch**.
   - The final report section must be titled "Case Coverage Checklist": provide a table of all cases and label each case's bottleneck group, applicable solution, and expected impact.
   - **Solution-merging rules**:
     - When multiple optimization directions involve **different templates with mutually exclusive branch conditions** (for example, AR path versus ARA path, or R ≤ threshold versus R > threshold), **merge them into one solution** (one kernel with multiple template branches) rather than splitting them into separate solutions.
     - When multiple directions involve **the same branch of the same template** (that is, different optimization strategies target the same group of cases), select the best strategies as parallel solutions for measurement and comparison in Step 2.
     - A merged solution's case-coverage table must identify which template branch each case follows.

[Stage 3 Outputs]
- Performance Tuning Plan: solution overview + concrete measures + Skill name and internal reference-file relative path + **per-case coverage checklist**.
- **The report must be written** to `{output_dir}/round{N}/performance_optimization_plan.md`.
- The performance optimization search-coverage record must be written to `{output_dir}/round{N}/optimization-search-coverage.json`; ordinary validation requires only structural and reference consistency.
- The report must contain the common `Status`, `Stage`, `Summary`, and `Details` fields.
- Explicitly label open-ended additions as "Not yet covered by the knowledge base".

---

[Acceptance Criteria]
- Complete the three stages in order, without skipping or reversing them.
- The Performance Tuning Plan may contain multiple viable solutions and must explicitly label coverage dimensions.
- Use thresholds and criteria from the skill files opened for this run.
- **Performance data has been collected for every case, and every case appears in the report.**
- Do not modify the operator source.
```

---

## Step 2: Solution Implementation

Step 2 is completed by **tilelang-tuning (the main agent)** in three stages: solution implementation, unified performance collection and reporting, and full correctness regression after final selection.

### Stage 2a: Parallel Solution Implementation (Main Agent → Multiple Implementation Subagents)

The main agent reads the Performance Tuning Plan, generates a unique safe identifier `sN_<slug>` for each solution, and then requests a separate `tilelang-perf-impl-expert` subagent for each solution, **running them in parallel**. Each instance receives exactly one solution and the complete message below:

> ⚠️ **Template first**: if the Performance Tuning Plan identifies a TileLang template `.py` file, pass the template file's complete path to the implementation expert in the prompt. When the implementation expert encounters a TileLang compilation error, it must first analyze the complete error and troubleshoot and fix it in the following order. It must not check only the common examples listed below.

```text
Implement the following single performance-tuning solution:

- Source-project root: {code_dir}
- Target operator file relative path: {operator_relpath} (relative to `{code_dir}`; the main agent later uses it to regenerate a profiling file from each optimized solution's latest source)
- Solution description: <the solution's complete content extracted from the Performance Tuning Plan, including template code path, key skeleton structure, template branch conditions, and per-case template mapping>
- Solution identifier: <unique safe identifier `sN_<slug>` generated by the main agent>
- Correctness test file: {test_file} (pytest correctness-test path relative to `{code_dir}`; use the test's built-in reference implementation and correctness thresholds)
- Correctness baseline report: {output_dir}/precision-baseline.md (the pytest file SHA256 within the current solution must match it)
- Performance-case file: {cases_csv} (CSV format, used only for performance-collection scope; **not a correctness oracle**)
- Output directory: {output_dir} (store optimized code and other artifacts under this directory. Store current-round artifacts in the `{output_dir}/round{N}/` subdirectory)
- Backend: {backend} (TILELANG_DEFAULT_TARGET={tilelang_target})
- Hardware-detection evidence: {hardware_evidence} (complete JSON for the same target device, including `full_soc`, `npu_arch`, and provenance)
- Framework source root: {repo_root} (use the confirmed root for the current repository and record it separately from the optimized-copy directory)

**Template-first instructions**:
- If the solution names a TileLang template `.py` path, copy the optimization pattern from that template into the target kernel's `.py` file, making only changes required to adapt it to the target operator.
  - Adaptations commonly include tile shape, buffer shape/dtype, SIMD parameters, and `pass_configs`, but are not limited to those items.
  - Identify any other required adaptations from the target kernel's interface, data layout, compute semantics, and actual compilation results; do not inspect only the examples above.
- Preserve the types and compute structure of native `T.simd.*` instructions in the template. Do not downgrade them directly to automatic `T.Parallel` expansion without validation.
  - Check actual adaptations involving target-buffer addresses, masks, repeat/stride, dtype, and boundary handling.
  - Those items are common areas to inspect, not an exhaustive checklist.
- Before modifying the implementation, load the Skill named `tilelang-performance-best-practices` and verify the TileLang APIs involved in the solution against invocation patterns that already compile and run in the current repository. Locate the framework source for the current repository, verify the actually imported version, and inspect API definitions, compilation constraints, `examples/ascend/`, and lowering for the selected backend. If that Skill is not installed or cannot be loaded by name, stop and report.
  - Determine API-check scope from APIs actually used by the solution; do not query only APIs or examples listed here.
  - For API usage that cannot be confirmed, continue inspecting corresponding documentation or repository implementations based on the actual code.
- On a TileLang compilation error, first read and analyze the complete error, determine the failing stage and direct trigger location, and then troubleshoot and attempt fixes in this order:
  1. Ascend hardware constraints and operator-instruction constraints
  2. Buffer shape, dtype, scope, lifetime, and version count
  3. TileLang API parameters, call structure, and lowering constraints
  4. Synchronization, memory planning, and other relevant `pass_configs`
  5. Adaptation between the template and the target kernel's interface, data layout, and boundary handling

  Common issues include GEMM missing `transpose_B=True`, L0C using a dtype other than float32, passing `threads=` to `T.Kernel` incorrectly, insufficient buffer version count, and mismatched `pass_configs`. These are examples only, not an exhaustive checklist.

  Every repair iteration must continue diagnosis based on the actual error. Even when none of the examples above applies, do not immediately simplify the optimization solution. Simplify only the affected part after troubleshooting based on the error, relevant API documentation, and existing runnable implementations still cannot fix it. Briefly list the actual checks, attempted fixes, and reasons for failure.

[Tasks]
Follow the execution flow defined for the agent whose logical name is `tilelang-perf-impl-expert` on the current host: copy directory → implement code → compile → verify correctness.
Directory naming: `{output_dir}/round{N}/optimized_<solution_identifier>/` (copy source from `{code_dir}` into this directory and modify it there).
Copy scope: follow the minimum-file-set rule under "Solution Implementation" in the definition associated with the `tilelang-perf-impl-expert` role. After copying, `{test_file}` must be runnable and debuggable within the optimized directory.
Test-file constraints: every solution uses an unmodified, content-identical copy of `{test_file}` in the current solution directory. Do not run the absolute path of a test file in the original repository and do not create a solution-specific pytest. If test coverage is insufficient, return to the main agent for a unified update that is then used for every solution. Before running, confirm that the target operator module loads from the current optimized-solution directory.
Correctness regression: reuse the devices, case list, and test configuration confirmed in `precision-baseline.md`. Run concurrently according to the actual available-device count only when device binding and isolation have been confirmed; otherwise, run serially. On OOM, reduce concurrency without dropping cases. Repair iterations may use `TILELANG_DEFAULT_TARGET={tilelang_target} pytest {test_file} -x`; final implementation acceptance must remove `-x` and cover the complete list.
Correctness-failure handling: fix implementation defects and rerun the complete target test suite. Only when evidence confirms that a small optimization point is affected by a framework issue, API bug, hardware limitation, or unsupported API may that optimization point be reverted while preserving the rest. Do not revert the whole solution to the old version. Return failure status if any FAIL remains.
On completion, return these common report fields: backend and actual target, optimized-code directory, compilation status, correctness details for the complete target test suite, pytest SHA256, absolute path and SHA256 of `{operator_relpath}` inside the optimized directory, **template-usage details**, and evidence for every localized revert. Do not collect performance, generate a performance report, or generate or reuse a profiling file.
```

The main agent waits for all subagents to finish. Before returning, each implementation expert is responsible for fixing compilation or correctness issues in its solution. A solution that still fails must record failing cases, root cause, and attempted fixes, then be excluded from Stage 2b. All other solutions that compile and pass correctness acceptance for the complete target test suite proceed to independent evaluation. Step 2 stops in the current round only if every solution fails.

### Stage 2b: Unified Performance Collection and Reporting (Executed by the Main Agent)

1. **Correctness confirmation**: accept only Stage 2a solutions that compiled successfully, passed complete-target-test-suite acceptance, and have a pytest SHA256 matching the baseline. Failed solutions must not have a profiling file generated and must not enter performance collection. If no solution succeeds, Step 2 fails and evaluation stops.
2. **Regenerate profiling files separately from every latest source (highest-priority hard gate)**:
   - Before collection, separately locate `{operator_relpath}` in the round's baseline and every successful Stage 2a `optimized_<solution>`, and regenerate a standalone profiling file from the **current latest byte content** of each source. Store them under `{output_dir}/round{N}/profiling_entry/<variant>/`.
   - Generate each new file according to Step 0: use the target source's complete bytes unchanged as a prefix, and make the host entry point run every case in `{cases_csv}` in the same order.
   - **Never reuse** a profiling file from Step 0, Step 1, a previous round, the baseline, or another optimized solution. Do not copy, rename, or patch an old file as a substitute for regenerating from the current solution's latest source. Regeneration is required even if the filename, kernel name, or solution directory is unchanged.
   - For each variant, record and verify the absolute source path and SHA256, profiling-file path, source-prefix SHA256, `{cases_csv}` SHA256, and generation time. If the source prefix is not byte-for-byte identical to the variant's latest source, or cases differ, stop Step 2b immediately; do not collect or compare.
3. **Unified msprof collection**: load the Skill named `tilelang-op-profiling`. For the baseline and every successful solution, use Step 2 of that Skill to collect all cases in a single run. If the Skill is not installed or cannot be loaded by name, stop and report.
   - **Do not select only representative cases**: all cases must be covered.
   - Set `--launch-count` to the number of cases. Do not restart msprof per case or change CASES order.
   - If the `tilelang-op-profiling` Skill takes a fallback path, it must still run the local kernel and every CASE from the newly generated file for that variant.
   - **Do not replace msprof collection with host-side timing (`std::chrono`, `gettimeofday`, and so on).**
   - After compilation, **verify a one-to-one correspondence between kernel-mangled names captured by msprof and kernel function definitions in the variant's latest source corresponding to that profiling file**. A mismatch means an old or incorrect kernel was measured; stop and regenerate immediately, without using the result.
   - Use aiv_time for AIV-only, aic_time for AIC-only, and the critical-path kernel time defined by the `tilelang-op-profiling` Skill for mixed AIC/AIV. Preserve the same timing basis for the baseline and all successful solutions of each case.
4. **Generate the Performance Tuning Report** and write it to `{output_dir}/round{N}/performance_optimization_report.md`:
   - Record `{backend}` and `TILELANG_DEFAULT_TARGET={tilelang_target}`; compare results only within the same backend.
   - **Per-case kernel-time comparison table** (msprof kernel_time): rows = all cases; columns = baseline + each successful solution. Label each case's timing basis.
   - **Per-case analysis**: explicitly state for every case:
     - The current performance bottleneck (bound type + key metrics)
     - The applied optimization technique (referencing the specific solution)
     - The speedup achieved (baseline kernel_time / optimized kernel_time)
   - Cases with the same bottleneck characteristics **may be grouped to describe the bottleneck and technique together**, but every case's speedup data **must be listed independently**.
   - Solution comparison table and improvement magnitude.
   - **Best-solution label**: select the successful solution with the greatest improvement by geometric-mean speedup across all cases and provide its code directory. If every successful solution has geometric-mean speedup ≤ 1, select the current round's baseline and explicitly record "No BEST_DIR / baseline selected".
   - Label the collection method and data paths.
   - Add a "Profiling File Provenance Gate" table: for every variant, list the latest source path/SHA256, profiling-file path, source-prefix SHA256, cases.csv SHA256, runtime kernel name, and all consistency conclusions.

[Acceptance Criteria]
- Every solution entering performance comparison compiled successfully and passed Stage 2a correctness acceptance for the complete target case set. Failed solutions are recorded and excluded; evaluation does not continue if every solution fails.
- The report contains an msprof kernel_time comparison table and per-case analysis (bottleneck → technique → speedup) for **every case**.
- Every case runs on the NPU.
- The baseline and every successful solution entering performance comparison use a profiling file freshly regenerated from and validated against their own latest source; there is no cross-solution or cross-round reuse.
- **Solution consistency**: implemented parameters match the original recommendations in the Performance Tuning Plan. If they differ, state why and include a control experiment using the originally recommended parameters.
- The report has been written to disk.

---

### Stage 2c: Full Correctness Regression After Final Selection (Executed by the Main Agent; Final Hard Gate)

After Stage 2b selects the **final solution** by geometric-mean speedup, run one full correctness regression on that implementation:

1. In the final-solution directory, or the selected-baseline directory, first confirm that `{test_file}` SHA256 matches `precision-baseline.md`, then run `TILELANG_DEFAULT_TARGET={tilelang_target} pytest {test_file}`. Final acceptance must not use `-x` and must produce complete per-case results.
2. Evaluate using the byte-identical pytest correctness logic from Step 0.5. Every case in the complete target list must pass; do not separately compare or relax numerical thresholds.
3. If a regression occurs, record failing cases, the complete pytest log, imported-source path, and SHA256; produce a failure report and stop the current round. The main agent does not modify or revert operator source. Stage 2a's implementation expert is already responsible for diagnosing, fixing, and locally reverting solution defects; do not establish a second code-implementation workflow during final verification.
4. Produce `{output_dir}/round{N}/full-precision-regression.md`, containing the common report fields, backend, actual target, pytest SHA256, and per-case PASS/FAIL.

[Gate] The round completes only if the full regression passes under `TILELANG_DEFAULT_TARGET={tilelang_target}` with the same pytest correctness logic.

---

## Step 2.5: Final Code Archival (Required)

### Preconditions

Execute this after a multi-round loop terminates, after a single-round Step 2 finishes, or after Step 1 determines that no optimization is needed. The main agent executes it directly.

### Scenario Identification

| Scenario | Determination | Validation |
|----------|---------------|------------|
| **A Original archive** | Code is produced directly by the Step 2 implementation expert, or the baseline is archived because no usable optimization exists | Controlled-source manifest validation |
| **B Restored archive** | Code is reimplemented/restored after code loss | Controlled-source manifest + no correctness regression + no performance regression |

> Scenario B must additionally run correctness- and performance-regression gates: even if reimplemented code compiles and is functionally bit-identical, it may regress under strict correctness criteria (MARE) or the performance measurement basis.

### Procedure

**Inputs**: `BEST_DIR` (best-solution code directory; may be empty), `ROUND_BASELINE_DIR` (input baseline for the round containing the best solution), `ORIGINAL_BASELINE_DIR` (original baseline, used only to record total improvement), `TEST_FILE` (unified pytest file), `CASES_CSV` (final performance-case file), `OPERATOR_RELPATH` (target operator file path relative to the project directory), `BACKEND`/`TILELANG_TARGET` (values selected at entry), `SCENE` (A or B)

**1. Select the archive source**: when `BEST_DIR` exists, set `SOURCE_DIR=BEST_DIR`. When Step 1 determines no optimization is needed, or every solution is slower than the baseline and there is no `BEST_DIR`, set `SOURCE_DIR=ROUND_BASELINE_DIR` and record the archive status as "Baseline; no usable optimization".

**2. Archive**: use a copy method that supports exclusion rules to copy `SOURCE_DIR` into `{output_dir}/final_optimized/`, excluding `.git/`, `operators/`, build, dist, `*.egg-info`, caches, profiling, logs, and report artifacts. Do not recursively copy the output directory.

**3. Controlled-source integrity validation** (A and B):
- Generate a "relative path + SHA256" manifest for controlled source and project configuration files in `SOURCE_DIR`, `final_optimized/`, and `ROUND_BASELINE_DIR`. Cover at least `.py/.h/.hpp/.c/.cc/.cpp/.cu/.json/.toml/.yaml/.yml`, `CMakeLists.txt`, and pytest configuration, excluding artifacts from Step 2.
- The `final_optimized/` manifest must exactly match `SOURCE_DIR`; otherwise, archival fails.
- When `BEST_DIR` exists, its manifest must differ from `ROUND_BASELINE_DIR` in at least one controlled source file. No difference means solution implementation failed.
- When `BEST_DIR` is empty, equality with the baseline is permitted, and the manifest must explicitly state that the baseline was archived and why.

**3b. Correctness no-regression gate** (B only): in `final_optimized/`, confirm that `TEST_FILE` SHA256 matches `precision-baseline.md`, then run `TILELANG_DEFAULT_TARGET={tilelang_target} pytest {TEST_FILE}`. Every case must pass under the same pytest correctness criteria. On regression, diagnose forward and fix, retrying at most three rounds.

**3c. Performance no-regression gate** (B only): first regenerate a Scenario-B-specific profiling file from the latest `OPERATOR_RELPATH` in `final_optimized/`, and pass the source-prefix SHA, `CASES_CSV` SHA, and runtime-kernel-name gates. Never reuse an old file. Then follow Step 2 of the loaded `tilelang-op-profiling` Skill to collect all performance cases once. Compare against the pre-loss report using identical cases and timing basis; require geometric-mean performance to be at least 95% of pre-loss performance and per-case regression to be at most 10%.

**4. Generate ARCHIVE_MANIFEST.md**: include the common report fields, backend, solution round, solution identifier or "Baseline", speedup, controlled-source manifest summary, Scenario A/B, archive time, source directory, current-round baseline directory, and original baseline directory. For Scenario B, attach the correctness- and performance-gate conclusions.

### Completion Criteria

- `final_optimized/` contains complete compilable source + `ARCHIVE_MANIFEST.md`.
- The archived controlled-source manifest matches the selected source directory. When `BEST_DIR` exists, the manifest must also differ from the baseline for the round containing that solution.
- Scenario B: correctness- and performance-regression gates pass.

### Key Requirement

- If Scenario B gates do not pass, do not archive; report the result accurately to the user.

---

## Common Report Format

Every stage report must contain the following fields for tilelang-tuning to parse. If any field is missing, treat the stage as incomplete:

```markdown
**Status**: ✅ Complete / ❌ Failed / ⏭️ Skipped

**Stage**: Step 0.5 Correctness Baseline / Step 1 Performance Data Collection and Analysis / Step 2 Solution Implementation / Step 2 Performance Report / Step 2 Full Correctness Regression / Step 2.5 Final Archive

**Summary**: One sentence describing the conclusion of this stage

**Details**: (stage-specific content)
```

### Per-Case Coverage Format (Applicable to Reports from Every Stage)

Whenever a report includes a case list, it must use the following format to ensure complete case coverage:

```markdown
| Case | Shape | dtype | Bottleneck Type | Applicable Solution | Timing Basis | Baseline kernel(us) | Optimized kernel(us) | Speedup | Notes |
|------|-------|-------|-----------------|---------------------|--------------|---------------------|----------------------|---------|-------|
| 1 | xxx | fp16 | VEC BOUND | Solution A | aiv_time | 14.1 | 6.0 | 2.35x | LUT optimization |
| 2 | xxx | fp32 | VEC BOUND | Solution B | aic_time | 37.6 | 26.7 | 1.41x | Tiling optimization |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
```
