---
name: tilelang-perf-optimize
description: Design performance-optimization plans for TileLang operators running on the PTO backend in the current repository. Use source, all target cases, PTO compilation results, and same-methodology profiling evidence to group cases, attribute bottlenecks, design Tiling and pipeline candidates, validate TileLang API and template feasibility, and produce up to three verifiable and reversible optimization plans. Use after performance data collection; do not use to modify operator code directly or handle ordinary functional failures.
---

# TileLang/PTO Operator Performance-Plan Analysis

This Skill targets only the TileLang frontend and PTO backend. Its goal is to convert measured bottlenecks into solutions that can be implemented and validated in TileLang code. It does not maintain APIs, code templates, or migration rules for other programming models.

## Scope of Responsibility

- This Skill is responsible for organizing baseline evidence, analyzing bottlenecks per case, grouping cases, designing Tiling and pipeline candidates, ranking solutions, and designing validation.
- The Skill named `tilelang-performance-best-practices` is responsible for TileLang APIs, execution domains, buffers, bundled Ascend implementation references, template maturity, PTO lowering, and implementation gates. It must be loaded when designing a plan; stop and report if it is not installed or cannot be loaded by name.
- The public interfaces, `examples/ascend/**/*.py`, and tests in the current repository are product sources of truth. The actually imported TileLang source and its `examples/ascend/` and `testing/ascend/` are frontend-capability sources of truth. Matching `references/*/code/*_asc.py` files in the loaded best-practices Skill are source-reading inputs for host dispatch, tiling, dataflow, buffer, and pipeline hypotheses.
- PTOAS receives the result of TileLang lowering. A capability expressible by the lower-level compiler does not imply that the TileLang frontend already exposes a corresponding API.
- This Skill does not modify kernels directly. When the user requests implementation, pass solutions that cleared the gates to the implementation stage.

## Case Input Gate

Establish the performance-case set before beginning any tuning analysis:

- Cases used during operator generation, accuracy validation, or system testing (ST) are not performance-tuning cases by default.
- Admit a case to the performance scope only when the user explicitly submits it as a performance case or confirms it for performance tuning. Its presence in tests, prior execution, or available profiling data does not count as user confirmation.
- If no performance cases have been explicitly submitted or confirmed, stop at this gate and ask the user to provide or confirm them. Do not silently substitute generation, accuracy-validation, or ST cases.
- Correctness cases remain regression inputs for validating an optimization, but they do not expand the performance-case set.

## Required Inputs

For the user-confirmed performance-case set, confirm that all of the following are available before analysis begins:

- The complete TileLang kernel to optimize, its dispatch/build entrypoint, and correctness tests.
- Every target case: case ID, shape, dtype, attributes, and public-interface constraints.
- The current code's accuracy baseline and PTO compilation status.
- Per-case kernel latency under the same device, environment, warmup, repeat, and concurrency settings.
- Available profiling metrics or traces; do not invent a bound type when metrics are unavailable.

If per-case latency, code-to-case mapping, or a correctly running baseline is missing, return to the data-collection stage without producing definitive optimization conclusions.

## Workflow

### 1. Fix Facts and the Comparison Methodology

For a public operator with multiple launches, first load the Skill named `tilelang-op-profiling` and follow the full-chain timing checks at its internal relative path `references/kernel-chain-measurement.md`. Verify kernel names, aggregation rules, and case-to-launch mappings to avoid planning against an undercounted baseline. Stop and report if the Skill is not installed or cannot be loaded by name.

1. Record the current commit, actually imported TileLang path, PTOAS version, device, and test command.
2. Reconstruct the kernel branch actually entered by every case from tests and dispatch; do not infer it from the operator name.
3. Verify that kernels in profiling originate from the target source, then organize baseline latency and available metrics for every case.
4. Use [Evidence and Bottleneck Attribution](references/evidence-and-diagnosis.md) to distinguish compute, data movement, scalar/scheduling, parallelism, and launch overhead.

### 2. Group by Execution Path

Group first by the actual computation path, then consider shape size. Recommended order:

1. Semantic path: elementwise, broadcast, reduction, transpose/gather, GEMM, attention, or irregular.
2. Dtype and accumulation precision.
3. Current dispatch/kernel branch.
4. Alignment, tails, empty tasks, and special attributes.
5. Measured bottleneck characteristics and shape scale.

Every case must appear in exactly one baseline bottleneck group. See [Tiling and Case Modeling](references/tiling-and-cases.md) for the grouping method.

### 3. Build a Resource Model of the Current Implementation

For every group, write down inspectable current values rather than starting from a fixed template:

- Independent task count, actual core count, tasks per core, and load-tail imbalance.
- Tile shape, loop count, valid element count, and padded footprint.
- GM/UB/L1/L0 buffer sizes, buffer versions, resident data, and safety margin.
- Bytes read and written per tile, useful computation, repeated movement, and intermediate-result lifetime.
- Logical dimensions on which output expressions depend; loop invariants and common subexpressions that do not depend on dimensions such as table, channel, head, or output column; and logical work before and after hoisting or reusing quantization operations.
- For irregular/SIMT paths, count integer multiplication/division/modulo, bit operations, casts, branches, index calculations, and live state per thread separately. Do not use FLOPs alone to represent actual computation cost.
- Actual use of `T.Persistent`, `T.SimdVF`, `T.SimtVF`, `T.gemm`, or reduction structures.
- Load the Skill named `npu-arch` and reuse the caller-provided `full_soc`, `npu_arch`, and complete `evidence`. If evidence is missing or incomplete, or the device or configuration has changed, have that Skill rerun its bundled detection script. Accept only consistent evidence for the Ascend950PR/DT family with `npu_arch=3510`, and record the complete model, actual core count, and physical capacity. Then load the Skill named `tilelang-performance-best-practices` and verify execution-domain reservations and the complete footprint against the current lowering. Stop and report if either Skill is not installed or cannot be loaded by name. Recalculate after any execution-domain or buffer change; do not reuse an arbitrary safety cap.
- For Elementwise, gather/scatter, or layout transformation, use the corresponding guide to build a nested task tree, tile boundaries, and the physical Vector-route cost for a fixed useful payload. Every feasible route that may improve the bottleneck must have a representative candidate or conclusive elimination evidence.
- For multistage pipelines, record the lifetime, versions, steady-state/tail range, and total footprint of every buffer live across iterations according to the relevant guide.

For every case, record `outer_items`, independent inner chunks, core count, outer/flattened task waves, payload per task, kernel time, movement/compute lower bounds, and main pipes. Analyze granularity boundaries, complete physical routes, and interaction combinations per case; planar/scratch materialization must not be recorded as directly register-resident. Structural directions not included in the final three solutions still require an adjudication basis. The invoking tuning workflow manages their exact statuses and on-disk format.

Provide specific numerical parameters only after the hardware capacity, alignment, and API behavior used by the formulas have been confirmed from current source or a verified implementation.

### 4. Generate and Screen Candidates

Internal mixed precision, HF32, and approximate instructions may be listed as experimental candidates, but they must not modify the reference implementation, tests, original tolerances, or public contract. Retain them only after correctness passes and same-methodology performance improves.

The optimization points, experience, and templates in this Skill are summarized or validated candidate sources and examples; they do not exhaust or close the TileLang operator optimization space. First generate candidates from the current operator source, all target cases, the resource model, and profiling evidence. Similar optimization principles from this Skill may be transferred, combined, or extended, and candidates not documented in this Skill may be proposed. Matching an existing template is not a stopping condition. Continue to examine non-template candidates that directly address measured bottlenecks and have TileLang API and PTO lowering support. Every candidate must have a falsifiable hypothesis and undergo subsequent accuracy and same-methodology performance validation. When evidence is insufficient, mark it `DESIGN_ONLY` according to the gates in this section.

Every candidate must correspond to a falsifiable hypothesis, for example:

- Hoist intermediate results that do not depend on an output dimension and reuse them in UB, eliminating repeated computation across table/channel/head.
- Increase or decrease tile size to reduce fixed overhead, increase parallelism, or control buffer pressure.
- Adjust task-to-core mapping to reduce load imbalance or idle cores.
- Use `T.Persistent`/`T.Pipelined` with complete buffer versions to improve movement/compute overlap; when automatic analysis is inapplicable, design manual versioning with the current API.
- Consume intermediate values within the same execution domain to reduce GM round trips or repeated reads.
- Preserve required fp32 state for reduction/GEMM and adjust hierarchical reduction or K-dimension pipelining.
- Split dispatch branches for alignment boundaries and tails so that every case does not pay the cost of a generic path.

Before implementation, rank candidates by removable work, match to the measured bottleneck, likelihood of closing the target gap, implementation risk, and validation cost. A reduction in logical work is only an upper bound on benefit; report actual speedup only from same-condition measurements. If multiple semantically equivalent arithmetic forms have no difference in lowering or generated IR, retain only one candidate.

Then load the Skill named `tilelang-performance-best-practices` and complete each of these gates:

1. Find evidence for APIs and invocation structure in a similar operator implementation in the current repository or in the actual TileLang source.
2. Read matching `references/*/code/*_asc.py` files as complete Ascend host-and-kernel examples. They require no maturity entry and may be used to form concrete hypotheses about dispatch, tiling, execution domains, data movement, buffer residency, fusion, and pipelines. Their presence alone does not prove that a strategy is faster for the current target cases.
3. When reusing a bundled executable template from that Skill, record the Skill name, internal relative template path, and maturity. Only `PRODUCTION_REFERENCE` or revalidated `VERIFIED` templates may be direct-reuse candidates. Real implementations in the current repository or actual TileLang source do not need prior registration in the status table, but their APIs, lowering, accuracy, and applicability must still be verified.
4. A bundled `EXECUTABLE_BASELINE` template is only a correctness starting point. `PARTIAL`, `DESIGN_ONLY`, or material without an executable kernel must not enter a directly implementable solution as a template. If a candidate has separate evidence from the current repository, actual TileLang source, or a concrete `_asc.py` implementation reference whose APIs are confirmed in current source, adjudicate it independently rather than requiring template registration.
5. If lowering, layout, or API parameters cannot be confirmed, mark the candidate `DESIGN_ONLY`; do not invent TileLang usage.
6. Pipeline candidates must read design guidance from `references/elementwise/double_buffer_design.md` at that internal relative path in the loaded `tilelang-performance-best-practices` Skill. Specify automatic/manual version selection, all versioned buffers, scheduler-visible control flow, generated-code checkpoints, criteria for effective operation on hardware, and execution-order stability tests. Writing only `num_stages=N` does not constitute an implementable solution.
7. Elementwise, gather/scatter, or layout-transformation candidates must read `references/elementwise/tiling_task_vector_search.md` and output Tiling boundaries, nested-task parallelism, and Vector-instruction cost. Every unselected high-value physical route requires elimination evidence.
8. When capacity affects Tiling, residency, or pipelining, use the loaded `npu-arch` Skill and complete evidence to confirm physical resources, then verify compiler reservations from the current TileLang lowering. Include a ledger with the complete model, physical resources, compiler reservations, explicit/resident/multiversioned footprint, and remaining capacity. Adding even a `SimtVF` used only for initialization or tails requires recalculating from the final kernel IR. Bandwidth or compute utilization must use the specification tier corresponding to the complete model; when it cannot be confirmed, do not substitute typical PR/DT values.

Retain at most three admitted solutions, whose status may only be `IMPLEMENTABLE` or `EXPERIMENT`. Whenever possible, each solution should validate only one primary hypothesis. Mutually exclusive dispatch branches may be combined into one solution, but branch conditions must be listed. List `DESIGN_ONLY` separately under "Non-Admitted Candidates" and explain missing API, lowering, or validation evidence; explain missing template evidence only when the candidate depends on a bundled reference. It does not count toward the three-solution limit and is not passed to implementation. The solution limit cannot delete structural obligations. If multiple orthogonal axes change one another's contiguous payload, amortization of fixed costs, or bottleneck, retain a combined solution or explicit infeasibility evidence.

### 5. Output Solutions and Validation Order

Following [Plan and Validation](references/plan-and-validation.md), output:

- A baseline-facts table and per-case bottleneck table.
- For every solution: evidence, hypothesis, concrete parameters, TileLang code-change locations, reference implementation, template maturity when applicable, and applicable cases.
- Accuracy, PTO lowering, per-case performance comparison, and rollback conditions.
- A complete case-coverage checklist; when no evidence supports optimization, explicitly write "No changes for now."

## Hard Rules

- Do not treat empirical thresholds as conclusions; the bound must be supported by data from the current run.
- Do not omit slow cases, tail cases, or currently failing cases, and do not use averages to hide a single-case regression.
- Do not pass by reducing reference precision, relaxing tolerances, removing tests, or short-circuiting on the Host.
- Do not present lower-level IR, internal compiler capabilities, or design documents as TileLang frontend APIs.
- Do not report "theoretically faster" as a performance gain; report speedup only from same-condition measurements.
- A candidate failure rejects only the exact recorded combination. Failure of automatic versioning, stage 2 slowing for one tile, accuracy failure for one gather-index format, planar-materialization fallback, or insufficient gain for a small group must not be generalized to claim that manual versioning, other tiles, direct register reorder, or complete contiguous units are all infeasible.
- Structural-coverage audits do not depend on whether the user supplies a numerical target. High-value obligations must enter an admitted solution or have elimination/inapplicability evidence. Evidence must not be declared converged while an unadjudicated route could reduce physical memory traffic or extra materialization for the primary bottleneck, improve effective lanes, fill core parallelism, or complete an admitted pipeline.
- Once a pipeline is admitted by per-core iteration count, capacity, and MTE/Compute overlap potential, automatic failure or partial effectiveness must be followed by adjudication of a representative explicit manual multiversion candidate. Close it only when automatic pipelining is fully effective or conclusive API/lowering/capacity/dependency evidence exists.
- "Strictly dominated" requires concrete instruction, memory-access, conversion, materialization, and synchronization costs for the same useful payload. Generic speculation cannot close a candidate.
- Do not modify source during plan analysis; implementation, compilation, and measurement occur in subsequent stages.

## Scenarios

- Compilation, accuracy, and profiling data are available for every case: map branches, group bottlenecks, and output up to three admitted solutions; list `DESIGN_ONLY` separately as non-admitted candidates.
- Only source or a small amount of latency data is available: output missing evidence and a collection plan; do not guess the bound, specific tile, or speedup.
- Only a lower-level compiler capability description exists, with no evidence for current TileLang expression and lowering: retain it as `DESIGN_ONLY` and do not pass it to direct implementation.
