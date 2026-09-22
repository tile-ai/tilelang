# TileLang Flash Performance Optimization Workflow

This mode assumes that the initial operator already passes correctness. It runs one full correctness test only after optimization is complete. The main agent directly performs analysis, implementation, and validation without invoking performance-analysis or solution-implementation subagents. The workflow imposes no default limit on duration, iteration count, or case count; it stops when the evidence converges.

## Inputs and Outputs

Required inputs: the entry-selected `{backend}`/`{tilelang_target}`, project root, target operator file, existing correctness pytest, complete parameters for performance cases supplied directly by the user, and `full_soc`, `npu_arch`, and complete `evidence` for the same target device. Cases used during operator generation, accuracy validation, or system testing (ST) are not performance cases by default. Only cases that the user explicitly submits or confirms for performance tuning satisfy the performance-case input requirement; their presence in source files, pytest, generated artifacts, previous runs, or profiling data does not count as confirmation. If hardware evidence is missing or incomplete, or if the device or configuration changes, reacquire it according to item 3 under "Establish the Baseline". If any other input is missing, ask only for that input. Do not generate a candidate table or infer cases. Before confirmation, do not create a tuning directory, compile or run code, perform profiling, or modify code.

Explicitly use `TILELANG_DEFAULT_TARGET={tilelang_target}` throughout the workflow: `pto` for PTO and `ascend` for AscendC. Defaulting to PTO, silently falling back, or comparing across backends is prohibited.

After the case input is confirmed, create an isolated optimization directory under `operators/{operator_name}[_timestamp]/` and perform the initial source copy using the unified copy scope below. The optimization directory contains:

- `working/`: the current best implementation;
- `candidates/Ci/`: an isolated candidate created from `working/`;
- `profiling/`: baseline and iterative performance data;
- `optimization-search-coverage.json`: per-case structural obligations and candidate events;
- `final_optimized/`: the code that passes final validation;
- `performance_optimization_report.md`: a concise record of the process and conclusions.

Do not modify the original project. Record SHA256 values for the original source and pytest. Do not modify the reference, correctness thresholds, or test cases to obtain a pass.
Create `timing.json` when execution begins and record the wall-clock start time. Record start, end, and elapsed time separately for the baseline, every candidate, and final acceptance. At completion, write total wall-clock time and summarize it in the final report. Timing is for retrospective analysis only; it is not a default stop condition or hard budget.

### Unified Copy Scope

The copy scope is fixed to the smallest file set that can run and debug the target operator inside the optimization directory:

- The target operator file and existing correctness pytest;
- In-repository modules imported directly or indirectly by either file;
- Required `__init__.py` and `conftest.py` files along those module paths;
- Required pytest/project configuration, such as `pytest.ini`, `pyproject.toml`, and `setup.cfg`.

Do not copy unrelated operators, the entire operator source directory, or the entire project root. If the output directory is inside the project root, exclude it to prevent recursive copying. The initial copy establishes the file manifest. Reuse that manifest directly when creating later `Ci` directories and archives; do not expand the copy scope again. Add files to the manifest only when compilation or testing proves that an in-repository dependency is missing, and record the reason.

## Priority of Facts

1. The current operator, pytest, public interface, and dispatch;
2. Measurements from the current round's compilation, correctness checks, and the Skill named `tilelang-op-profiling`;
3. The actually imported TileLang source, lowering for the selected backend, and similar implementations in the repository;
4. Material from Skills such as `tilelang-performance-best-practices`.

Lower-priority experience must not override higher-priority measurements. Skills propose candidates; they do not directly prove benefit.

## Default Implementation Contract (Temporary Single-Kernel Gate)

Unless the user explicitly permits multiple kernels, the target operator must satisfy the following constraints by default:

- A target factory defines only one nested `@T.prim_func` that performs the operator's work, and all cases resolve to the same logical runtime kernel. Do not split fast paths, slow paths, or different shapes into multiple primfuncs and dispatch among them on the host.
- Do not add a performance branch that matches an exact shape, count, or attribute value solely to target a selected performance case. Branching inside the same primfunc is permitted when based on interface semantics and general properties, such as capacity, alignment, dtype, task count, full/tail tiles, contiguity, or hardware resource limits.
- Python factory parameters and `T.macro` may organize code. After macro expansion, the code must still belong to the same kernel; macros must not bypass the constraints above.
- Different shapes may use a small number of JIT parameter combinations from the same kernel definition, such as different tiles or core counts. The launcher may perform only a short configuration selection and must not duplicate the algorithm implementation.
- When establishing the baseline, record the target primfunc count, dispatch conditions, and runtime kernel name. Before profiling every candidate, repeat the source/AST audit and verify the runtime kernel name. Reject candidates that violate the contract; performance gains do not qualify them for promotion.

This gate restricts case specialization and multi-kernel dispatch. It does not prohibit selecting different execution paths within the same kernel based on general, provable conditions.

## State

- `B0`: the immutable original baseline;
- `B*`: the current validated best version, initially `B0`;
- `Ci`: the current-round candidate generated from `B*`.

Compute both values in every round:

```text
Incremental speedup = B* kernel_time / Ci kernel_time
Cumulative speedup = B0 kernel_time / Ci kernel_time
```

## Workflow

### 0. Case Input Gate

1. Build the performance-case set only from cases that the user explicitly submits as performance cases or explicitly confirms for performance tuning.
2. Treat every case whose only provenance is operator generation, accuracy validation, or ST as unconfirmed. Do not use such a case by default, even when it is already present in source files, pytest, generated artifacts, prior execution records, or profiling data.
3. Validate the explicitly submitted or confirmed case parameters. If they are missing, ask for them and stop; selecting Flash mode or asking for performance tuning does not itself confirm any existing case.
4. Do not generate a candidate table or automatically select, supplement, or group cases from pytest or a benchmark.
5. After user confirmation, lock the list and record its SHA256. Any change requires reconfirmation. Do not continue until this gate passes.

### 0.5 Historical-Evidence and Validated-Candidate Reuse Gate

Before establishing the baseline, load the Skill named `tilelang-performance-best-practices`. Then inspect, read-only, the current git history, existing same-operator artifacts under `operators/`, candidates in the current worktree, and `VERIFIED`/`PRODUCTION_REFERENCE` records in that Skill to avoid repeating exploration of structures that have already been measured. If the Skill is not installed or cannot be loaded by name, stop and report.

For a `PRODUCTION_REFERENCE`, compare the current kernel and launcher against the feature fingerprint in the corresponding documentation. If core structures are missing, treat the local file as a baseline; do not inherit performance conclusions from the path or status table alone.

At minimum, the feature fingerprint must cover the data-residency lifecycle, local/GM layout, intermediate materialization, synchronization points, tile/core configuration, static tail blocks, buffer versions, and launcher JIT parameters. Expand semantic labels such as "fusion", "reuse", "merge partials", and "pipeline" into physical buffers, GM copies, reduction lowering, and version configuration. Identical names do not mean a structure has been reproduced. Rank optimization points independently according to the current bottleneck; do not inherit performance evidence for a whole group when only some features have been reproduced.

A historical candidate may enter the current candidate pool only when its public interface, target cases, test methodology, TileLang version, selected-backend version, and device architecture are compatible. If any critical item is incompatible, use it only as a design clue. Record the candidate source, source SHA, inherited performance evidence, and incompatibilities. Do not describe inherited results as having been independently discovered from `B0`.

Rank compatible historical candidates and new candidates together by bottleneck match, upper-bound benefit, evidence maturity, and validation cost. Do not automatically prioritize them or require reproducing them as a group. A historical result cannot be promoted directly to `B*`; it must still pass compilation, correctness, and same-method profiling in the current environment under this workflow.

### 1. Establish the Baseline

1. Confirm that the Case Input Gate has passed. Then read the target operator, pytest, call chain, similar implementations, actually imported TileLang source, and lowering for the selected backend.
2. Confirm that the baseline compiles and runs correctly. By default, the initial operator is assumed to have passed correctness, so do not repeat the full correctness test.
3. Before proposing a performance hypothesis, load the Skill named `npu-arch` and reuse the entry-provided `full_soc`, `npu_arch`, and complete `evidence`. If the evidence is missing or incomplete, or if the device or configuration has changed, use that Skill's bundled detection script to reacquire it. Accept only consistent evidence for the Ascend950PR/DT family with `npu_arch=3510`. Record the complete model, physical UB/L1/L0, and actual core count. From the loaded `tilelang-performance-best-practices` Skill, read the resource-discovery rules at the internal relative path `references/common/hardware_resource_discovery.md`. Verify execution-domain reservations in the final kernel body against actual TileLang lowering and establish a ledger of explicit, resident, and multi-version footprints. Recompute it after adding or removing `SimtVF`, buffer versions, padding, LUT/index data, or temporary layouts. Do not replace effective capacity with an unsourced fixed cap. Bandwidth and theoretical compute must match the complete model's specification tier; if this cannot be confirmed, do not compute exact utilization. If `npu-arch` is not installed or cannot be loaded by name, stop and report.
4. Establish a source semantic-cost model: list the logical dimensions on which every output expression depends, such as layer, token, prefix, table, and channel. Count integer multiplication/division/modulo, XOR, casts, branches, and index calculations per valid element or token. Examine loop invariants, common subexpressions across output dimensions, repeated transfers, intermediate-value lifetimes, and per-thread live state. Quantify the logical work that hoisting or reuse could eliminate. This is only an upper bound on candidate benefit, not a measured speedup.
   If considering multistage pipelining, also draw the `CopyIn → Compute → CopyOut` read/write dependencies of every mutable UB/L1 buffer between adjacent iterations. Mark the version count, effective iterations per core, full-wave range, and tail range. A design that versions only inputs while omitting outputs or temporary buffers still read asynchronously must not proceed to implementation.
   For Elementwise, gather/scatter, or layout conversion, also read the search rules at the internal relative path `references/elementwise/tiling_task_vector_search.md` from the loaded `tilelang-performance-best-practices` Skill. Draw the outer-item/inner-chunk work tree; enumerate complete contiguous-unit, SIMD, DMA, capacity, and parallelism boundaries. For a fixed amount of valid output, compare implementable dataflow families by contiguous/scattered access, register operations, temporary materialization, and lane utilization. Do not assume that two specific API paths are the predetermined answer.
5. Generate a standalone profiling entry point from the latest `B0` kernel and launcher, embedding all target cases. The host entry point must reproduce the launcher's parameter handling and invocation method while calling the local identical kernel. Validate source SHAs, actual parameters, cases, and runtime kernel name.
6. Load the Skill named `tilelang-op-profiling`, collect ordinary metrics for all target cases, and generate `summary.txt`. If the Skill is not installed or cannot be loaded by name, stop and report.
7. Cross-check the semantic-cost model against measurements to determine whether the bottleneck is compute, transfer, scalar/scheduling, parallelism, launch, or mixed. Do not attribute a specific source operation as the primary bottleneck from a single pipe ratio alone; attribution requires evidence from source workload, generated IR/instructions, or additional profiling.
8. Before the first modification, read the complete [Performance Optimization Search Coverage Gate](optimization-search-coverage/optimization_search_coverage_gate.md). Create `{output_dir}/optimization-search-coverage.json` (schema v3) from the [starter template](optimization-search-coverage/optimization_search_coverage.example.json): record per-case structural facts, targets, lower bounds, and pending adjudication obligations. Append candidate `CREATED`/`RESULT` events to `candidate_events`; a candidate's identity and applicable cases must not change meaning midway. Run `python "$(git rev-parse --show-toplevel)/agent/core/scripts/validate_optimization_search_coverage.py" {output_dir}/optimization-search-coverage.json` successfully before creating a candidate.

### 1.5 Candidate Pool and Benefit Ranking

Before the first modification, produce at most four mutually distinguishable active candidates. Update the execution queue in each later round using new data. Candidates come from current source and profiling analysis, Skill structures already read, compatible historical evidence, and verified similar implementations. Different provenance neither separates pools nor determines promotion. Prioritize algorithmic repeated-computation elimination, data residency and task mapping, Tiling/pipelining, and arithmetic lowering backed by generated-code evidence when they match the current bottleneck. Do not fill a quota in every category. The four-item cap limits only the current execution queue, not coverage obligations; a pending item must not disappear because it falls out of the top four, ranking changes, or successive candidates have low benefit.

For an applicable Skill or historical candidate, compare source, buffers/copies, lowering, and launcher both when proposing and after implementing it. Count it as implemented only when the physical structure matches. The directions above are candidate examples, not a closed list; matching a historical structure or existing template does not justify ending exploration.

Use append-only `candidate_events` in `optimization-search-coverage.json` as the closed-loop ledger. A separate readable candidate index may be created, but it must not become a second source of truth for state. Every initial or subsequently reopened high-benefit candidate must have a `CREATED` event and corresponding `RESULT`, with the result recorded as `PROMOTED`, `REJECTED_WITH_EVIDENCE`, `INAPPLICABLE_WITH_EVIDENCE`, or `BLOCKED_BY_ENVIRONMENT` according to the gate. Reordering the candidate pool must not delete entries. While the target is unmet, ranking changes or environmental failures must not disguise unresolved obligations as closed.

Candidate events are mandatory on-disk execution records for every round, not report content backfilled at the end. When a candidate is created, fix its six-axis identity `(work granularity, task mapping, physical dataflow, precision, storage/pipeline, tail strategy)`, per-case granularity facts, structured physical route, pipeline facts, and applicable cases, then append `CREATED`. Before creating `Ci+1`, append a `RESULT` with the same identity and applicable cases. Record, per case, whether the modified path actually executed; then record source SHA, compilation/correctness/performance results, bottleneck shifts, status, and next ranking, and run the validator. Any material change to an axis, granularity value, route kind, or applicable path receives a new ID. Do not reuse an old ID with a new meaning or batch-backfill from memory at the end.

For every candidate, record a falsifiable hypothesis, applicable cases, estimated logical work eliminated, upper-bound performance benefit, implementation/correctness risk, TileLang and selected-backend basis, validation cost, and rollback point. Calculate the current gap to the user's acceptance metric. For example, when the metric is inversely proportional to time and the data volume is fixed, calculate the maximum elapsed time and remaining speedup required to reach the target. Candidates whose estimated upper bound cannot cover a significant target gap may be diagnostic items only; do not repeatedly prioritize them over candidates that could cover the gap.

Rank candidates by measured-bottleneck match, eliminable work, probability of covering the target gap, maturity of reuse evidence, and implementation/validation cost. If several source expressions are expected to be semantically equivalent, first compare lowering, generated IR, or key instruction structure for the selected backend. If there is no material difference, merge them into one candidate; do not spend multiple on-device validation rounds on spelling-only rewrites.

For Elementwise, gather/scatter, or layout conversion, decompose candidates by `tile/group × task mapping × Vector dataflow × precision × pipeline`. If one axis changes another axis's contiguous payload, fixed-cost amortization, or primary bottleneck, add or reopen the combination as a high-benefit candidate. Failure of tiling alone and failure of SIMD alone do not replace measured or definitive infeasibility evidence for their combination; adjudicate combinations according to the tiling/task/Vector guide's matrix.

Cover Vector dataflow candidates by their complete physical route from source to result, not by source-level names or by filling a quota based on the execution domain of the final arithmetic. Materializing a complete planar/transpose/scratch representation before Vector arithmetic is a materialized transform; it does not close the "direct load of the original contiguous window + register select/shuffle/pack" route. While the target remains unmet, every physical family in the route plan that is still relevant to the bottleneck must either be measured per case or rejected with definitive evidence. As long as an unresolved route can reduce per-lane UB/GM accesses, full-materialization passes, or significantly improve the active-lane ratio, add or reopen a candidate event.

A route candidate must use the lowest-cost representative of its physical family supported by current API/lowering evidence. Its record must list dtype/lane/part, implicit or explicit widening, select/shuffle/interleave/pack, arithmetic, and the output store/scatter chain. Failure of an initial implementation that still contains removable repeated widening, lane repair, or intermediate materialization rejects only that complete chain, not the entire physical family. Mark it and reopen a representative candidate according to the Vector guide. Across route comparisons, keep tile/group, task mapping, precision, output path, and stage count fixed whenever possible. If they cannot be fixed, do not attribute the overall candidate time to one intrinsic.

Write `route_experiment` into every candidate's structured physical route. If `source_window.register_window_feasible=true` and the direct route is not closed by the baseline or by definitive API/lowering/same-payload cost evidence, then the first candidate with `route_experiment=true` must be `CONTIGUOUS_LOAD_REGISTER_REORDER`. Candidates that change only other axes while inheriting the parent version's physical route are exempt from this ordering. Before implementation, use the Vector guide's `vld + vselr` template to verify index scope, dtype, predicate, and cross-register segmentation. Do not alter the ordering merely because a gather example is more familiar.

Adjudicate work-granularity candidates against actual source values for complete contiguous-unit, SIMD, DMA, capacity, and parallelism boundaries. Measuring a pair or arbitrary small group closes only that value. If a case has both granularity and physical-dataflow obligations, the same candidate must measure `FULL_CONTIGUOUS_UNIT × CONTIGUOUS_LOAD_REGISTER_REORDER`, or provide precise infeasibility evidence. Separate failures along the two axes do not replace adjudication of their combination.

When tile, core count, stage, static tail blocks, and buffer versions interact, treat them as a configuration family and screen a small number of evidence-backed combinations under the same kernel definition. First compare them using the same performance method, then fully validate the winning combination. Lack of benefit for one combination rejects only that combination, not the entire structural direction.

Before admitting a pipeline candidate, read the complete design document at the internal relative path `references/elementwise/double_buffer_design.md` from the loaded `tilelang-performance-best-practices` Skill, and explicitly answer: automatic or manual multiversioning; why lowering can recognize it; which buffers need versions; how control flow exposes a stable full-wave pipeline; how tails are handled; the pipelines expected to overlap; and activation/failure criteria. If these answers are incomplete, treat it only as a diagnostic experiment.

Once a pipeline candidate is admitted based on iteration count, capacity, and MTE/Compute overlap potential, automatic and manual implementations form a complete decision chain independent of the numerical performance target. If automatic buffer versions are ineligible, extent/alias lowering is incorrect, dynamic stage intrinsics fail, only some live buffers are versioned, or latency/overlap does not activate, reject only the precise automatic combination. Continue adjudicating explicit input/output/temporary storage, manual annotations, unconditional full-wave execution, a static stage body if required, `T.Pipelined`, and a structurally matched stage-1 counterpart. Close the pipeline obligation only when the automatic combination is fully effective or when the manual path has definitive infeasibility evidence from current APIs, lowering, capacity, or dependencies.

### 2. Single-Hypothesis Iteration

Validate only one primary hypothesis per round. When there is a clear dataflow, layout, or lowering dependency, combine the required changes into one candidate and state the dependency; do not mix unrelated optimizations:

1. Select the currently highest-ranked hypothesis from the candidate pool. Using the existing file manifest, create `candidates/Ci/` from the current `B*` in `working/`. Do not continue stacking changes from an unpromoted or regressed candidate, and do not modify `working/` directly. Create combination candidates independently from `B*` as well.
2. Compile and validate the shape being tuned in this round. If validation fails, fix or reject the candidate; do not relax correctness. During iteration, do not run a targeted correctness test or the full correctness test suite for the relevant complete test set.
3. Regenerate and validate a profiling entry point from the latest `Ci` kernel and launcher. Follow Steps 2–3 of the loaded `tilelang-op-profiling` Skill to collect and archive current-round data. If `msprof op` fails only because instrumentation/export fails, and direct execution of the same latest entry point succeeds, use ordinary `msprof` according to Step 2.5 of that Skill. Do not reject a candidate or declare convergence before the fallback is complete. If results are anomalous, directionless, or close to noise, analyze the complete CSV and PipeTimeline.
4. Compare `Ci`, `B*`, and `B0` on the same device with identical inputs, warmup, launch count, concurrency, and timing basis. Do not substitute host timing for kernel data.
5. Read the new `summary.txt` and required raw data. Confirm whether total time, the original bottleneck, and inter-core balance improved and whether a new bottleneck appeared. Use this evidence to promote or reject the candidate, or select the next hypothesis, and update the ranking and upper-bound benefit of all remaining candidates.
6. Before creating the next candidate directory, update `optimization-search-coverage.json` and pass ordinary validation. If a bottleneck shift gives another orthogonal axis new combined benefit, add or reopen the combination obligation and candidate, then rerank it. Pipeline conclusions from an old structure do not carry over after work granularity, task mapping, or physical dataflow changes. Rerun pipeline admission for cases that still have overlap-capable iterations.

A pytest benchmark is for quick screening only; it cannot replace Steps 3–5 as evidence for a new bottleneck attribution. While the target remains unmet, divide remaining paths into fixed segments per launch/core and hot loops repeated per tile/element. Modify the relevant segment only after confirming attribution from cross-case absolute time, timeline, lowering, or key instructions.

While the performance target remains unmet, attribution to scheduling or hardware limits requires timeline, lowering, or key-instruction evidence. Do not conclude from wave count, pipe ratio, or Task/block time differences alone.

A pipeline candidate must also inspect version switching and synchronization structures in generated code/IR, as well as same-method kernel latency, effective bandwidth, and relevant pipe overlap. Two UB buffers, successful compilation, successful execution, or `num_stages > 1` alone does not prove that the pipeline activated. If neither latency nor overlap improves beyond noise, classify it as ineffective and continue checking control flow and dependencies rather than retaining a merely formal double buffer.

Before every profiling round, audit the "Default Implementation Contract" and write the primfunc count, basis for general branches, and resolved target kernel name into the candidate record.

### 3. Promotion Decision

First repeat collection of the same version to estimate current-environment noise. When a candidate difference is close to noise, alternate measurements as `B* → Ci → B* → Ci` and decide using a stable mean or median.

Set `B* = Ci` only when all of the following conditions hold:

- Compilation and the correctness gate for the shape tuned in the current round pass;
- Aggregate performance improves over `B*` by more than measurement noise plus the required safety margin;
- There is no unacceptable regression in any single case or unmodified path;
- The benefit is reproducible and hits the target kernel;
- All target cases for both `B*` and `Ci` were collected with the same method using the loaded `tilelang-op-profiling` Skill. A pytest benchmark alone cannot justify promotion.

By default, evaluate overall benefit using the geometric mean across all target cases and also report the worst case and regressed cases. When the user specifies weights or a performance target, use the user's objective. If a case tradeoff cannot be adjudicated, do not overwrite `B*`; retain the candidate and report it.

If a candidate is not promoted, restore `B*` and record the hypothesis, source SHA, tests, performance results, and failure reason. Do not continue stacking changes on a regressed version without a clear dependency.

Only when a candidate is promoted may `candidates/Ci/` update `working/` using the existing file manifest. Unpromoted candidates remain in their own directories and do not change `working/`. After changes to relevant input/reuse, accumulator/reduction, partial/copy/host behavior, index placement, tile/core/stage, tail blocks, or buffer lifetime, an old negative conclusion rejects only its original combination. Reassess the direction in the new context.

Before promoting a candidate involving multiversion buffers, pipeline synchronization, persistent scheduling, or compiler warnings, cover all target cases within the same process and repeat measurement at least once using formal warmup/repeat settings. If different cases use different general paths, repeat again in reverse or interleaved order. A warning neither automatically rejects nor admits a candidate: promote after recording it only when results are stable and correctness and performance are reproducible. Reject the candidate and retain diagnostic evidence if elapsed time drifts abnormally with case order, intermittent errors occur, or results are unstable across runs.

### 3.5 Search Escalation and Deduplication

- After an arithmetic- or syntax-lowering candidate fails to improve beyond noise, stop equivalent rewrites of that class unless generated IR or key instructions prove another spelling is materially different.
- If two consecutive candidates each improve by less than 3% while the target remains unmet, pause implementation and reexamine output dependency dimensions, repeated computation, per-thread live state, task mapping, and historically validated structures. The next candidate must come from algorithmic reuse, dataflow, or parallel structure with a higher upper-bound benefit, or the report must explicitly state that no such implementable candidate exists.
- While the current target still requires more than 1.2× speedup, do not consecutively implement local candidates whose benefit can only be explained as low single-digit percentages, unless one is a necessary prerequisite for validating a later high-benefit solution.
- Every candidate-pool reorder retains rejected hypotheses and generated-code evidence. Do not retest the same lowering under different source spelling.
- If a slow case has fewer outer items than available cores, first adjudicate flattening independent inner round/chunk work. Keeping a serial fallback is not evidence of optimization.
- If task-granularity changes alter contiguous payload, source-window density, fixed-cost amortization, or the primary bottleneck, create a new interaction obligation with physical dataflow/pipelining. Dataflow failure under the old granularity, or failure of the new granularity paired with the old dataflow, does not replace adjudication of their combination.
- To close a candidate as "strictly dominated", attach concrete instruction, memory-hierarchy, conversion, materialization, synchronization, and lane costs for the same effective payload. Code complexity or speculation about more instructions cannot close a high-benefit obligation.

### 4. Stop Conditions

Stop iteration when any condition is met:

- The user's performance target or explicit resource budget is reached;
- Performance is near a verifiable theoretical hardware lower bound;
- Successive candidates have not improved beyond noise and there is no stronger evidence for a new hypothesis;
- Remaining directions lack an implementation basis for the currently selected backend;
- Further optimization would break correctness or coverage, or cause an unacceptable regression.

Before stopping, run `--final` according to the [Search Coverage Gate](optimization-search-coverage/optimization_search_coverage_gate.md). If every case reaches the user's target, obtain `TARGET_MET`. If the target remains unmet, `SEARCH_INCOMPLETE` from ordinary `--final` does not authorize stopping; continue the search. Only after per-case structural coverage and verifiable lower bounds are closed may `--final --allow-unmet-convergence` return `UNMET_BUT_CONVERGED` and permit declaring evidence convergence. Device, profiler, or toolchain failure is recorded only as `BLOCKED` and is not structural closure evidence. Report explicit user-budget exhaustion or environmental blockage accurately as incomplete; do not call it convergence. A geometric mean or fast case must not hide a slow case. To claim that a hot path is already the minimum sequence, also provide the minimum semantic operations, generated instruction/memory-access counts, relevant API search scope, and adjudication of alternative routes.

If the user's target is not met, the report must list unmeasured high-benefit candidates and why they were not implemented. An unmet target or environmental blockage must not be reported as target completion.

### 5. Final Acceptance

1. After all performance tuning is complete, run `env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET={tilelang_target} python -m pytest {test_file}` on `B*` to complete one full target test suite. Device access and concurrency follow workflow-entry conventions. If it fails, final acceptance fails and that candidate is not delivered.
2. Generate separate profiling entry points from the latest source of `B0` and `B*`, then remeasure all target cases with the same method using the loaded `tilelang-op-profiling` Skill. Recollect the baseline if the environment drifted.
3. If the relevant complete test suite is fully green and final measurements improve on `B0`, archive `B*` to `final_optimized/` using the existing file manifest from "Unified Copy Scope". Otherwise, archive the baseline with the same manifest or retain only incomplete candidates.
4. Generate `performance_optimization_report.md`: status, backend, environment and SHAs, final comparison for all cases, worst regression, key iterations, stop reason, complete wall-clock duration, and artifact paths.

The report provides both implementation status and target status. Implementation status uses only:

- `VERIFIED`: the relevant complete test suite passes and same-method measurements improve. This means only that the implementation is validated, not that the user's performance target is met;
- `NO_CHANGE`: there is no reliable benefit, so retain the baseline;
- `CANDIDATE`: the implementation has potential, but the final gates are incomplete;
- `BLOCKED`: blocked by the environment, device, compiler, or tests.

Target status uses only `MET`, `CONVERGED_WITHOUT_NUMERIC_TARGET`, `NOT_MET_CONVERGED`, or `NOT_MET_UNFINISHED`. Write `MET` only when the user's explicit target is reached. When the user supplied no numerical target and per-case structural coverage and the `--final` gate pass, write `CONVERGED_WITHOUT_NUMERIC_TARGET`. When a target exists but is unmet, write `NOT_MET_CONVERGED` only if the structural audit passes; otherwise, write `NOT_MET_UNFINISHED`. Do not use `VERIFIED` or geometric-mean improvement to conceal gaps against the target or in slow cases.

Without complete performance evidence, report only a hypothesis or candidate; do not claim that optimization is complete.
