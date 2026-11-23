---
name: tilelang-perf-tuning
description: "Performance tuning for TileLang operators that already have a correct implementation on Ascend950: lock the user's cases, establish device-side baselines, iterate candidates by evidence, and complete precision, performance, and search-coverage acceptance."
---

# TileLang Performance Tuning

The platform is fixed to Ascend950; the backend is passed in explicitly by the caller: PTO uses `TILELANG_DEFAULT_TARGET=pto` and AscendC uses `TILELANG_DEFAULT_TARGET=ascend`. Keep the same backend throughout the workflow and never hardcode a target in source files.

## Entry Gate

Before starting, obtain:

- The project root, the target operator file, the public wrapper/dispatch, and the existing precision tests;
- Complete performance case parameters directly confirmed by the user, including shapes, dtypes, attributes, and optional numeric targets;
- Evidence that the current implementation already passes precision;
- The physical device, the actually imported TileLang path, and the target backend.

When a case is incomplete, only request clarification; never generate, trim, or regroup cases from pytest, benchmarks, or public metadata on your own. After confirmation, write the cases into `cases.csv` or an equivalent structured file and record its SHA256; any case change requires renewed confirmation.

## Isolation Directory and State

Create an isolation directory at a user-specified location or under `operators/<op>_<timestamp>/` inside the project, without modifying the original project. Copy only the minimal file set needed to run the operator and tests independently; once fixed, all candidates reuse this list.

```text
working/                         current best B*
candidates/Ci/                  candidates created independently from B*
profiling/                      B0, candidate, and final collections
final_optimized/                delivered code after final acceptance
optimization-search-coverage.json
perf-report.md
timing.json
```

- `B0`: the immutable original baseline.
- `B*`: the current verified best, initially B0.
- `Ci`: created only from the current B*; never stacked on top of an unpromoted candidate.

While a candidate is not promoted, `working/` must not be overwritten; after promotion, update B* strictly following the initial file list.

## Workflow

### 1. Establish Facts and Baseline

Read [Performance Core](references/performance-core.md) in full. Read the operator, the wrapper, the tests, the actual dispatch, the current TileLang API/lowering, and similar implementations in the same repository; never infer the execution path from the operator name or memory.

Generate a standalone profiling entry from the latest B0 source with all locked cases embedded, and collect device-side baselines for all cases following the [Profiling Contract](references/profiling-contract.md). Record the source SHA, case SHA, kernel names, launch mapping, device, frequency, warmup, and formal run counts.

Establish per case:

- The actual execution path, task tree, tiles/cores, tails, and buffer lifetimes;
- Logical GM bytes, compute volume, integer indexing/branches/casts/synchronization, and fixed overhead;
- Hardware physical capacities, the effective capacities of the current lowering, and explicit/resident/multi-version footprints;
- Kernel latency, dominant pipes, inter-core tail imbalance, and verifiable lower bounds.

Before the first source modification, create and validate `optimization-search-coverage.json` following the [Coverage Ledger](references/coverage-ledger.md).

### 2. Route Candidate Knowledge

Read according to the current bottleneck instead of loading every reference by default:

- Elementwise, gather/scatter, transpose, layout, quant, or irregular memory access: read [Vector and Memory Routes](references/vector-memory-routes.md).
- Reduction/Norm/Softmax, GEMM/Quant, Scan/Recurrent, Attention, or RoPE: read only the matching sections of [Operator Patterns](references/operator-patterns.md).
- Multiple kernels or mixed AIC/AIV: also read the whole-chain timing section of the [Profiling Contract](references/profiling-contract.md).

Historical candidates, templates, and results from other devices only serve to propose hypotheses. A candidate may be promoted only after it is compatible with the current interfaces, versions, backend, data paths, and cases, and has been recompiled, precision-checked, and profiled under the same methodology in the current environment.

### 3. Build the Candidate Pool

Before the first modification, keep at most four active candidates; re-rank later based on new evidence. The priorities are:

1. Eliminate duplicated algorithmic work, GM round trips, or full intermediate materialization;
2. Improve contiguous transfers, on-chip residency, task mapping, and load balance;
3. Adjust tiles/cores, Vector/Cube dataflow, and precision paths;
4. Test pipelining only when eligible;
5. Consider low-yield local syntactic or instruction substitutions last.

For each candidate, record the falsifiable hypothesis, applicable cases, six-axis identity, the amount of work expected to be eliminated, the upper bound of the gain, API/lowering basis, capacity, precision risk, validation cost, and rollback point. The six axes are: work granularity, task mapping, physical dataflow, precision, storage/pipeline, and tails. Any substantive change on an axis gets a new candidate ID.

### 4. Single-Hypothesis Iteration

Each round:

1. Create an independent `Ci` from B* and implement exactly one primary hypothesis; necessary layout, synchronization, or lowering dependencies may land as part of the same dataflow change, but unrelated optimizations must not be mixed in.
2. Compile and run the targeted precision cases actually covered by the candidate; on failure, fix or eliminate the candidate — never loosen tolerances, modify the reference, or skip cases.
3. Regenerate the profiling entry from the latest Ci source and collect data for all target cases under the same methodology.
4. Compare Ci against B* and B0, checking total time, per-case regressions, the original bottleneck, inter-core balance, and new bottlenecks.
5. Append the coverage `RESULT`, update best metrics, the ranking of remaining candidates, and newly triggered structural obligations; run the normal validation before creating the next candidate.

When performance is close to noise, alternate re-measurements as `B* → Ci → B* → Ci`. Promote a candidate only when precision passes, all cases have been collected under the same methodology, the overall gain exceeds noise, and there is no unacceptable per-case regression. pytest/host benchmarks can be used for quick screening but cannot support final promotion or bottleneck attribution.

### 5. Stopping and Final Acceptance

Stop only when one of the following holds: the user's target is reached; the result approaches an evidenced hardware/algorithmic lower bound; consecutive candidates yield no above-noise gains and no stronger hypothesis exists; the remaining high-yield routes are definitively blocked by the current API/lowering/capacity/semantics; or further optimization would break precision or coverage.

Before stopping, run coverage `--final`. When the target is not met, declare evidence-based convergence with `--final --allow-unmet-convergence` only after all structural obligations and per-case lower bounds are closed. Device or profiler failures can only be recorded as `BLOCKED`; they are not evidence of convergence. Convergence requires a recorded resolution for every environment-blocked case, and the original blocked RESULT stays in the event log.

Finally:

1. Run the existing relevant full test suite once against B*;
2. Regenerate the entries from the latest B0 and B* sources and re-measure all cases under the same methodology;
3. Update the mutable case metrics and `best_metric_evidence` to reference the final measurement; when it differs from the promoted value, record it in the case's `final_measurement` (`implementation_id`, `kernel_time_us`, `artifact`) instead of editing the immutable `RESULT` event, then rerun the applicable final gate; an earlier `--final` run does not validate a later measurement, so a missed target after re-measurement reopens the search or the convergence evidence;
4. Archive B* into `final_optimized/` only when the full suite is green and the final measurements still beat B0 overall **and** satisfy the per-case acceptance criteria used for promotion (no case regresses beyond the allowed limit); otherwise deliver B0 or explicitly mark the result as an unfinished candidate;
5. Generate `perf-report.md` covering implementation status, target status, all case comparisons, the worst regression, key iterations, the stopping reason, unimplemented high-yield directions, environment/SHA/commands, wall-clock time, and artifact paths.

Implementation status uses only `VERIFIED`, `NO_CHANGE`, `CANDIDATE`, `BLOCKED`; target status uses only `MET`, `CONVERGED_WITHOUT_NUMERIC_TARGET`, `NOT_MET_CONVERGED`, `NOT_MET_UNFINISHED`.

## NPU Command Constraints

Every command that accesses the Ascend NPU explicitly uses:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<pto|ascend> <command>
```

When a specific device is required, select it with the entry's `--device <physical_id>`. Perform one device health check first; known faulty devices must not be used for candidate attribution. When the test project has an explicit device-binding convention, keep its correct methodology and explain the difference in the report.

## Scripts

- `scripts/perf_summary.py`: archives the eight `msprof op` CSV categories and generates a summary for the target kernel.
- `scripts/validate_optimization_search_coverage.py`: validates the normal coverage record and the final stopping gate.

The scripts locate themselves from the real directory of this skill and do not depend on the current working directory.
