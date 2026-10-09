# Performance Optimization Search-Coverage Gate

This gate prevents a performance search from recording only "candidates already considered" while omitting high-value paths directly triggered by case structure. It does not prescribe the optimization answer; it separates per-case facts, obligations requiring adjudication, and actual candidates.

## 1. Create the Record Before Modifying Source Code

Create an `optimization-search-coverage.json` with `schema_version: 3` in the tuning directory. It contains three append-only collections:

- `cases`: a structural diagnostic card for each performance case;
- `obligations`: coverage obligations triggered by structural facts that must be adjudicated before stopping;
- `candidate_events`: candidate `CREATED`/`RESULT` events.

For every case, record `outer_items`, `inner_independent_chunks_per_item`, whether the two are independent, `available_cores`, both task-wave counts, `per_task_payload_bytes`, baseline `kernel_time_us`, movement/compute lower bounds and their sources, the main pipes, `source_window`, and associated obligations. The user goal must also be recorded as `performance_target`, and the current best result measured with the same methodology must be written to `best_metrics`/`best_metric_evidence`; update them together on every promotion. A target may combine multiple `LE`/`GE` conditions with `ANY`/`ALL`, for example, "latency no greater than 10 us or semantic GM utilization at least 80%." If a lower bound is temporarily unavailable, use `null` and explain the missing evidence; guessing numbers is prohibited.

In addition to the overall obligation status, every associated case must have `case_dispositions`. A case counts as tested only when a candidate actually activates the modified path and the corresponding `case_results` entry in its RESULT is `TESTED_PATH` with a same-methodology kernel `performance_measurement`. Passing only compilation/accuracy, testing another shape, an inactive runtime branch, or another path from the same factory cannot substitute for testing that case.

The minimal skeleton follows; validator constants and errors define the complete enumeration values:

```json
{
  "schema_version": 3,
  "cases": [{
    "case_id": "x", "outer_items": 1,
    "inner_independent_chunks_per_item": 1,
    "inner_chunks_independent": false, "available_cores": 1,
    "outer_task_waves": 1.0, "flattened_task_waves": 1.0,
    "per_task_payload_bytes": 1, "kernel_time_us": 1.0,
    "lower_bounds_us": {"movement": null, "compute": null},
    "lower_bound_evidence": "...", "main_pipes": ["..."],
    "performance_target": {"logic": "ANY", "conditions": [
      {"metric": "kernel_time_us", "operator": "LE", "threshold": 10.0}
    ]},
    "best_metrics": {"kernel_time_us": 20.0},
    "best_metric_evidence": ["..."],
    "source_window": {
      "access_pattern": "INDEXED_OR_REORDERED", "bounded": true,
      "span_bytes": 128, "consumed_bytes": 96, "density": 0.75,
      "window_vregs": 2, "register_window_feasible": true,
      "window_scope": "smallest repeatable hot-loop source window",
      "decomposition_checked": true,
      "baseline_route_kind": "SCALAR_OR_SIMT",
      "baseline_writeback_kind": "SCALAR",
      "index_regularity": "periodic", "evidence": "..."
    },
    "obligation_ids": ["O1"]
  }],
  "obligations": [{
    "obligation_id": "O1", "kind": "...", "case_ids": ["x"],
    "status": "PENDING", "candidate_ids": [],
    "case_dispositions": [{
      "case_id": "x", "status": "PENDING", "candidate_ids": []
    }]
  }],
  "candidate_events": [{
    "candidate_id": "C1", "event": "CREATED",
    "identity": {
      "work_granularity": "...",
      "granularity_facts": {
        "unit": "...", "boundary_kinds": ["SIMD"],
        "values_by_case": {"x": 1}
      },
      "task_mapping": "...",
      "physical_dataflow": {
        "description": "...", "route_kind": "SCALAR_OR_SIMT",
        "source_access": "SCALAR", "writeback_kind": "SCALAR",
        "full_payload_materialization": false,
        "register_lane_reorder": false, "indexed_lane_access": false,
        "route_experiment": false
      },
      "precision": "...", "storage_pipeline": "...",
      "pipeline_facts": {"stages": 1, "realization": "SINGLE_STAGE"},
      "tail_strategy": "..."
    },
    "applicable_cases": ["x"]
  }]
}
```

A reusable pending template is available at [optimization_search_coverage.example.json](optimization_search_coverage.example.json), and a complete RESULT shape is available at [optimization_search_coverage.final.example.json](optimization_search_coverage.final.example.json). The example values demonstrate only the schema and must be replaced with facts and evidence from the current operator.

When a candidate finishes, append a RESULT with the same identity/`applicable_cases`, and fill in `result_status`, overall `evidence`, and `case_results` for every applicable case. `TESTED_PATH` must also provide a `performance_measurement` containing `kernel_time_us` and a profiling `artifact`; if accuracy has passed but performance has not yet been collected, still record `NOT_RUN_WITH_EVIDENCE`. A compilation failure must likewise be recorded as `NOT_RUN_WITH_EVIDENCE` for each case and must not be disguised as measured coverage. If `REJECTED_WITH_EVIDENCE` has no `TESTED_PATH`, structured `rejection_evidence` is also required: `failure_stage` (compilation, kernel execution, accuracy, or profiling validity), `observed_error`, a healthy-environment/baseline `control_check`, and a `candidate_finding` attributable only to that candidate. Whenever `rejection_evidence` is provided, it must satisfy this structure; free-form stage names cannot be used to bypass validation. Without a candidate-attribution control, the result cannot be marked rejected and must instead be marked blocked.

If the device, profiling service, external resource, or environment toolchain fails before a candidate obtains valid evidence, the candidate cannot be recorded as a performance or implementation failure, and failure text cannot close a search obligation. Use `BLOCKED_BY_ENVIRONMENT` for the candidate RESULT, `NOT_RUN_BLOCKED` for unexecuted cases, and record structured `blocking_evidence`: `kind`, `failure_stage`, `observed_error`, the performed `control_check`, and a verifiable `resume_condition`. The associated obligation, per-case disposition, and plan item remain `BLOCKED` and reference this candidate. Ordinary validation permits preserving this state; `--final` must reject it.

Only reproduction or a control check in a healthy environment can attribute a failure to the candidate itself. For example, if a candidate consistently triggers a kernel compilation/execution error under direct invocation while the baseline and device health checks succeed, that may count as candidate failure. If the device is unavailable before the candidate starts, a known-good entry also cannot launch, or the profiler's own instrumentation/export fails, it is only an environment block. After recovery, retry the blocked candidate and update the obligation status only after obtaining a valid result.

```json
{
  "candidate_id": "C3", "event": "RESULT",
  "result_status": "BLOCKED_BY_ENVIRONMENT",
  "blocking_evidence": {
    "kind": "DEVICE_UNAVAILABLE",
    "failure_stage": "before kernel launch",
    "observed_error": "device retain failed",
    "control_check": "known-good entry also cannot start the device",
    "resume_condition": "device health check passes, then retry C3"
  },
  "case_results": [{
    "case_id": "x", "disposition": "NOT_RUN_BLOCKED",
    "evidence": ["No candidate kernel launch occurred."]
  }]
}
```

The example above must still retain the exact same `identity` and `applicable_cases` as CREATED; unchanged fields are omitted here only for brevity. Set the associated obligation/plan `status` to `BLOCKED`, reference `C3` from `candidate_ids`, and record `evidence`; the obligation additionally records a string list in `blocking_reasons`.

Apply at least these general triggers:

1. Fewer than one wave of outer tasks with multiple independent inner chunks: create `inner_parallel_flatten`;
2. Adjacent combinable work, contiguous boundaries, or substantial fixed scheduling/memory-access costs: create `work_granularity_search`;
3. Per-element/per-lane noncontiguous memory access on the hot path with a bounded required source window: create `physical_dataflow_routes`;
4. The same case requires both work-granularity and dataflow adjudication: create `granularity_dataflow_interaction`;
5. Sufficient independent iterations per core with overlap opportunity in both MTE and Compute: create `pipeline_realization`.

Triggers only create items requiring adjudication; they do not guarantee that a solution will be faster. The maximum of four candidates applies only to the current execution queue and cannot limit or delete coverage obligations.

## 2. Work-Granularity Obligations

`work_granularity_search` must have a `boundary_plan`. For every associated case, record and adjudicate each of the following separately:

- `FULL_CONTIGUOUS_UNIT`: the largest contiguous unit that does not cross semantic/metadata boundaries;
- `SIMD`: integer boundaries for effective lanes/loads/selects/scatters;
- `DMA`: burst, complete 2D row, alignment, or stride boundaries;
- `CAPACITY`: effective capacity boundaries accounting for all resident, temporary, and multiversioned storage;
- `PARALLEL`: task boundaries for one wave, two waves, and steady-state multiple waves.

Each plan item records its derived values in `case_values`. A TESTED candidate's `granularity_facts` must contain the same boundary kind and per-case values. If an item is inapplicable or eliminated, provide evidence from the current layout, capacity, lowering, or same-payload cost. Testing one small group covers only that value; it cannot cover a complete dynamic segment with different values.

## 3. Physical-Dataflow Obligations

`physical_dataflow_routes` must have a `route_plan` and separately adjudicate four physical families for every associated case:

- `SCALAR_OR_SIMT`;
- `MEMORY_INDEXED_VECTOR`;
- `CONTIGUOUS_LOAD_REGISTER_REORDER`;
- `MATERIALIZED_TRANSFORM`.

"Direct register reorder" is strictly defined as: perform a contiguous Vector load from the original or already segmented source window, then select/shuffle/pack within registers without writing the fully reordered payload as an intermediate UB/L1 layout. A path that first writes planar data, transpose scratch, or another complete temporary layout and then performs a contiguous load for computation must be labeled `MATERIALIZED_TRANSFORM`; it cannot be labeled a direct route merely because the final arithmetic occurs in Vector registers.

A `route_plan` item is not a generic "input route"; it is an exact `route_kind × writeback_kind × cases` combination. The input side must distinguish at least the four families above, while the output side must distinguish at least scalar, contiguous, strided, indexed scatter, materialized copy, and mixed. A TESTED item accepts only a candidate whose input route and output writeback both match. Failure of one gather-index format cannot close the direct-register route, and a scatter/synchronization failure cannot close the same input route paired with other writeback methods.

While the target remains unmet, input/output combinations that still match the bottleneck must remain `PENDING` or be adjudicated using a measured candidate or hard infeasibility evidence. The gate does not require mechanical enumeration of the entire Cartesian product. However, if a candidate improves one axis but ultimately fails because of another, retain the improved axis and create at least one independent alternative combination for the dominant failing axis; do not close the entire structural family together.

Every candidate must also record `physical_dataflow.route_experiment`: it is `false` when only granularity/tasks/pipelining change and the physical route is inherited completely from the parent version; it is `true` when a gather, register reorder, complete materialization, or other route is added or replaced. For a case with `register_window_feasible=true` whose direct route has not yet been closed by the baseline or conclusive evidence, the first candidate with `route_experiment=true` must use `CONTIGUOUS_LOAD_REGISTER_REORDER`. This ordering gate constrains only physical-route experiments; it does not prevent first testing an independent task-mapping or granularity candidate.

`source_window` turns routing conditions into machine facts: access type, boundedness, span, bytes actually consumed, density, required register count, register-window feasibility, baseline input/writeback routes, index regularity, and evidence. The window here must be the "smallest repeatable hot-loop window required to generate one Vector output chunk," not an entire patch with row-stride holes, an entire row group, or a complete semantic segment. If processing can stream row by row or chunk by chunk, calculate span, density, and register count for the decomposed window, and write `decomposition_checked=true`. `INDEXED_OR_REORDERED` automatically triggers a physical-route obligation.

`register_window_feasible=true` does not mean "it might theoretically fit." It means that an implementable representative has been identified after accounting for dtype, single-/multi-register selection range, live-register pressure, and the current PTO lowering. Setting it to `false` requires `register_infeasibility` with the same structured capacity/lowering/semantic hard evidence used to close an obligation, proving that even the smallest repeatable window is infeasible. If this remains unknown, keep the direct route pending and investigate the API/lowering first; the size of a complete segment or stride holes cannot be used to bypass the ordering gate.

`route_experiment` states whether that candidate's `route_kind × writeback_kind` differs from the corresponding case's `baseline_route_kind × baseline_writeback_kind`. The validator cross-checks it; it is not a label the agent may assign freely. While an implementable direct-register route remains open, the first physical-route candidate that differs from the baseline must test it.

## 4. Granularity and Dataflow Interaction

When the same case creates both work-granularity and physical-dataflow obligations, a `granularity_dataflow_interaction` must be created. Every interaction item explicitly records `boundary_kind × route_kind × writeback_kind`. A high-value combination must either be tested by the same candidate that actually activates it, remain pending, or be adjudicated with the hard infeasibility evidence defined in the next section.

Separate results for `group + SIMT` and `baseline group + register Vector` cannot substitute for the combination. If the combination changes buffer lifetimes or the number of steady-state iterations per core, pipeline eligibility must also be reevaluated; a pipeline failure on the old granularity/dataflow invalidates only the old structure.

## 5. Candidate Identity and Closure Evidence

Candidate identity still consists of six axes: work granularity, task mapping, physical dataflow, precision, storage/pipeline, and tail strategy. `granularity_facts` contains the machine facts for the first axis, while structured `physical_dataflow` records both the input `route_kind` and output `writeback_kind`. Allocate a new ID whenever any axis, per-case granularity value, input route, or output writeback changes; IDs must not be reused with altered meaning. A fix that changes the copy route, intermediate materialization, synchronization/dependencies, buffer lifetime, or scheduling method is also a new candidate. Only a fix for a syntax or implementation defect that does not change the meaning of the six axes may reuse an ID.

A candidate failure inherently rejects only its own exact identity. `CLOSED_WITH_EVIDENCE` cannot be established with explanatory text; it requires structured `hard_infeasibility` and must declare `exact_scope_only=true`. The only allowed hard evidence is:

- `CAPACITY`: provide exact capacity and the complete storage footprint, and confirm that streaming/chunking still cannot decompose it;
- `LOWERING`: provide a minimal reproducer, the actual queried symbol, and compilation/lowering artifacts;
- `SEMANTIC`: provide the constraint and proof that semantics cannot be preserved;
- `STRICT_COST_DOMINANCE`: quantify the reference and candidate costs in the same units over the same effective payload; candidate cost must be strictly higher.

Long source code, implementation difficulty, API-search text, failure of one adjacent combination, a malformed ring, or one candidate with no gain is not hard evidence for closing other combinations in the same family. These facts may only remain in the RESULT for that exact candidate, after which the next combination is reprioritized according to the new bottleneck.

"Strictly dominated" must compare concrete read/write hierarchy, instruction/conversion/materialization counts, and feasibility over the same effective payload. An obligation cannot be closed merely because the source is longer, the instruction sequence appears larger, one adjacent combination failed, or a candidate is inconvenient to implement.

## 6. Pipeline Obligations

Once a pipeline has been admitted by structure and profiling, completeness is independent of the numerical performance target:

- If automatic versioning completely covers every buffer live across iterations and stage-1/2 pairing is proven effective, `automatic.status` may be recorded as `COMPLETE` with a reference to the actually tested candidate;
- Automatic ineligibility, lowering failure, only partial buffer versioning, or ineffective latency/overlap rejects only that automatic combination. Next, create a manual input/output ping-pong candidate, and change the full-wave/static-stage structure afterward if necessary. `manual.status=TESTED` must be bound to a MANUAL multistage candidate that actually ran the target case; a compilation or synchronization-ring failure does not count as tested;
- Even when automation is complete, explicitly record the manual path as `NOT_NEEDED_AUTO_COMPLETE`; it must not remain pending;
- After work granularity, task mapping, or physical dataflow changes, the original pipeline conclusion is not inherited automatically. Create a new obligation or retest cases that remain eligible.

Candidates also use `pipeline_facts` to record the stage count and `SINGLE_STAGE`/`AUTOMATIC`/`MANUAL`. Both `pipeline_plan.automatic` and `pipeline_plan.manual` are objects containing `status` and `candidate_ids`. A pipeline obligation uses `basis_candidate_id` to bind the structure it adjudicates; use `B0` for the initial structure. If a new single-stage candidate is promoted on an existing pipeline case, the final gate requires a new pipeline obligation based on that candidate; candidates that already form a complete multistage pipeline do not trigger recursion.

## 7. Validation and Stopping

Run ordinary validation before the first modification and before creating the next candidate. It checks only record structure and reference consistency, outputs `VALID_RECORD`, and allows `PENDING`; it is not a closure gate for candidate implementation. Run `--final` only when preparing to end tuning:

```bash
python "$(git rev-parse --show-toplevel)/agent/core/scripts/validate_optimization_search_coverage.py" \
  <optimization_search_coverage_json>

python "$(git rev-parse --show-toplevel)/agent/core/scripts/validate_optimization_search_coverage.py" \
  <optimization_search_coverage_json> --final
```

Ordinary `--final` outputs `TARGET_MET` only when the current best metrics for every case meet their targets. Once the target is met, unexplored routes may remain `PENDING`, but every candidate already created must have a RESULT; the gate does not force the agent to invent closure reasons for routes that no longer need exploration.

When the target is unmet, ordinary `--final` outputs `SEARCH_INCOMPLETE`. This is not authorization to stop: when there is no explicit user resource limit or external block, return to candidate analysis/implementation rather than proceeding to final acceptance, archiving, or termination. Only when further progress is genuinely impossible and the intent is to declare "evidence convergence" may the following command additionally be run:

```bash
python "$(git rev-parse --show-toplevel)/agent/core/scripts/validate_optimization_search_coverage.py" \
  <optimization_search_coverage_json> --final --allow-unmet-convergence
```

This exception path requires every obligation, per-case disposition, route/boundary/interaction plan, and pipeline chain to be closed by actual testing or the hard infeasibility evidence above. It also requires `unmet_convergence_evidence.case_bounds` for every case that misses its target: current best kernel time, a verifiable lower bound, evidence, and an explanation of the remaining gap. On success, it outputs `UNMET_BUT_CONVERGED`. Any `PENDING`, `BLOCKED`, unexecuted candidate, or textual inference causes failure. Machine validation proves only record consistency; detecting omitted high-value combinations still requires a structural audit against the source, per-case lower-bound gaps, and generated code.
