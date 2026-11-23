# Search Coverage Ledger

`optimization-search-coverage.json` prevents the candidate pool from recording only the approaches already thought of while missing high-yield routes directly triggered by case structure. Machine validation only proves the record is consistent; a structural audit still requires the source and per-case lower bounds.

## Creation Timing and File

Before the first source modification, create a schema v3 record from `coverage-example.json`; see `coverage-final-example.json` for a complete RESULT. Example values must be replaced with current facts.

The record contains three append-only collections:

- `cases`: per-case structure, baseline, target, and current best;
- `obligations`: obligations triggered by structural facts that must be adjudicated before stopping;
- `candidate_events`: candidate `CREATED` and `RESULT` events.

Each case records at least the outer/inner independent tasks, available cores, task waves, payload, kernel time, transfer/compute lower bounds, dominant pipes, the source window, the user target, and the best metrics. When there is no numeric target, `performance_target` is `null`; a baseline or an example threshold must never masquerade as the user's target.

## General Obligation Triggers

1. The outer dimension is smaller than one wave and the inner chunks are independent: `inner_parallel_flatten`;
2. Adjacent work, contiguous boundaries, or significant fixed cost exist: `work_granularity_search`;
3. The hot path accesses elements/lanes discretely and the window can be bounded: `physical_dataflow_routes`;
4. The same case needs both granularity and route adjudication: `granularity_dataflow_interaction`;
5. Each core has enough iterations and both MTE/Compute can overlap: `pipeline_realization`.

Triggers create obligations; they do not promise the method is faster. The at-most-four active candidate limit never deletes or closes obligations.

Pipeline-realization evidence must be measured on the obligation's current basis. `basis_candidate_id` is either one basis for every linked case or an object mapping each linked case to its own basis, so cases promoted on different candidates keep separate evidence instead of blocking closure. The validator binds each evidence candidate to its per-case basis from the promotion timeline: a candidate basis accepts evidence created after that basis's `PROMOTED` result and before the next promotion that activates the case (the basis itself stays eligible); a B0 basis accepts only evidence created before the first promotion that activates the case. Evidence must pair with the basis on `work_granularity`, `precision`, `task_mapping`, and the structured granularity facts — `granularity_facts.unit` plus the case's entry in `granularity_facts.values_by_case` — so a measured gain cannot include a tile-size change. For every case measured on B0, `basis_identity` records the same five facts; use the flat form when all linked cases share one B0 identity, or a per-case map keyed by the B0 cases otherwise. A candidate basis needs a `PROMOTED` result before it can carry evidence. Because a promotion closes the previous basis's interval for the cases it activates, a promoted single-stage candidate supersedes any basis that was promoted before it: rebase that case to the promoted candidate (or a newer basis) and re-measure. Normal validation may record a proposed basis that is not yet promoted only while the obligation stays pending; at the final gate the basis must be promoted at or after the superseding promotion, so a rejected candidate cannot discharge the rebase. Do not add a second obligation for a case that already has one, and do not close a superseded basis with stale results.

The work-granularity obligation records per case the source values for `FULL_CONTIGUOUS_UNIT`, `SIMD`, `DMA`, `CAPACITY`, and `PARALLEL`. The physical routes adjudicate at least the implementable combinations of `SCALAR_OR_SIMT`, `MEMORY_INDEXED_VECTOR`, `CONTIGUOUS_LOAD_REGISTER_REORDER`, and `MATERIALIZED_TRANSFORM` relevant to the current bottleneck, distinguishing output writeback.

## Candidate Events

`CREATED` fixes the six-axis identity, per-case granularity, physical route, pipeline facts, applicable cases, hypothesis, and gain upper bound. Append a `RESULT` with the same identity and applicable cases before creating the next candidate.

RESULT status uses `PROMOTED`, `REJECTED_WITH_EVIDENCE`, `INAPPLICABLE_WITH_EVIDENCE`, or `BLOCKED_BY_ENVIRONMENT`. Cases whose modified path actually ran use `TESTED_PATH` with `kernel_time_us` and a profiling artifact. Cases that never ran, were not activated, or only passed precision must not masquerade as performance coverage.

When a candidate fails compilation/execution/precision while a healthy B0 control runs normally, structured rejection evidence may attribute the failure to the candidate. Device, profiler, or external-environment failures can only be recorded as blocked, with the recovery conditions written down; the obligation stays blocked and the final gate rejects convergence. Once the environment recovers, record the resolution in the convergence evidence for that case (`blocked_resolutions`: the blocked `candidate_id`, the recovery description, and evidence) and keep the original `BLOCKED_BY_ENVIRONMENT` RESULT so a retry clears the block without rewriting history.

## Closing Evidence

A candidate failure only vetoes the exact six-axis combination. Closing an entire direction requires `hard_infeasibility` with `exact_scope_only=true`:

- `CAPACITY`: the full footprint plus proof that streaming/chunking cannot continue;
- `LOWERING`: a minimal reproduction, the actually queried symbol, and the compilation/lowering artifacts;
- `SEMANTIC`: proof that the public semantics cannot be preserved;
- `STRICT_COST_DOMINANCE`: quantified instruction, memory-access, conversion, materialization, and synchronization costs under the same effective payload.

Implementation difficulty, long source files, one adjacent combination failing, or a no-gain result are not closing evidence.

## Validation and Stopping

Run the normal validation before creating each next candidate:

```bash
python3 <skill_dir>/scripts/validate_optimization_search_coverage.py <record>
```

The normal validation prints `VALID_RECORD` and allows `PENDING`; it does not permit stopping. Before stopping, run `<record> --final`: it prints `TARGET_MET` when all targets exist and are met; `CONVERGED_WITHOUT_NUMERIC_TARGET` when no numeric targets exist, the structure is closed, and per-case bounds exist; and `SEARCH_INCOMPLETE` when targets are unmet.

Only when all obligations, per-case dispositions, route/boundary/interaction plans, pipeline chains, and the lower bounds of unmet cases are closed, run:

```bash
python3 <skill_dir>/scripts/validate_optimization_search_coverage.py \
  <record> --final --allow-unmet-convergence
```

Success prints `UNMET_BUT_CONVERGED`. Any pending item, blocked item, candidate without a RESULT, unresolved environment block, or textual inference fails.
