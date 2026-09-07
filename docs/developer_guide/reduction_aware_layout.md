# Reduction-aware layout selection

The opt-in `reduction-aware` policy searches vector widths at reducer-update
roots and accounts for the communication their layouts introduce:

```python
kernel = tilelang.compile(
    program,
    target="cuda",
    pass_configs={"tl.layout_cost_model": "reduction-aware"},
)
```

The default remains `register-count`. The `io-aware` policy is unchanged.
Components without reducer finalization retain register-count scoring.
Non-CUDA targets and functions with unknown serial trip counts also fall
back to register-count, rather than applying CUDA synchronization costs.

The default already explores native and scalar reducer plans. This policy
adds intermediate widths and communication-sensitive scoring; it does not
assume that avoiding communication always improves on the default. For
example, a `[4, 256]` column reduction with 128 threads can use width two
without a collective, rather than choosing between width four with a
collective and width one without one.

## Search and constraints

`GetVectorizeSize` remains a legality query. The search uses the existing
partitioner's adjusted legal upper bound and tries successively halved widths,
ending at one. Each width starts an isolated component-inference attempt.
Equivalent root layouts are discarded before propagating the component.
There is no Cartesian product of widths across update loops.

Explicit `coalesced_width`, loop layouts, and annotated reducer
`PartialFragment` layouts remain authoritative. Native candidates run first;
same-root ties retain the native plan and equal-cost roots retain program order.

## Physical plans, not inferred communication alone

The scorer's read-only analysis and `ReducerPlanAndMaterialize` share the
same physical-plan selection logic.
It includes destination containment, copy-only destination overrides, and
packed accumulation. In particular, a narrow `PartialFragment` can still
lower to a FullParticipant plan when its destination is incompatible. The
scorer charges the resulting full-participant collective, not the narrower
`CombineSteps` recorded in the inferred layout.

The policy does not extend reducer-index legality. For example, direct-memory
parallel nests can be fused before reducer analysis; index projections that
the existing ownership proof cannot handle retain the wide fallback.

Communication is charged per finalize execution, not per update site.
Constant enclosing serial trip counts weight updates and finalizations
independently. Batched finalization shares barriers across values while
retaining per-value combine, shuffle, and workspace traffic.
Batch-size constraints are checked only when the physical plan emits a
finalize operation; a seedless, local-complete plan removes that operation.

## Cost units

Measurable attempts are ordered by:

1. Estimated register-array spill traffic.
2. Execution cost in issue-equivalent bytes.
3. Per-thread register slots.

Unmeasurable attempts never beat measurable ones merely because an estimate
is unavailable. If every estimate is unavailable, spill/register ordering
still provides a deterministic fallback.

The memory term reuses the io-aware global-memory model, bounded below by
the actual update loop's memory issues, and includes shared load/store issue
estimates. The execution score is:

```text
reducer_issues = local_issues + collective_shared_issues + combine_issues
                 + 4 * shuffle_issues + 32 * barriers
execution = sum(participants * issue_lane_bytes * reducer_issues)
              + sum(operator_repeats * (global_cost + shared_cost))
```

`issue_lane_bytes` is `MaxVectorLoadBits(target, false) / 8`, currently 16.
This is a normalization against the memory model, not actual bytes moved by
arithmetic instructions. Spill traffic remains a separate, higher-priority
score field. Update and finalize repetition counts are already included in
`reducer_issues`; operator repetition weights the ordinary memory accesses.

Local updates are analyzed as materialized read-modify-write stores after
loop partitioning. Their legal vector width is distinct from arithmetic
packing: a vectorized fp32 load does not imply a four-lane fp32 add.
Reduction-axis stores remain scalar unless the physical plan introduces
packed accumulation lanes.

The initial CUDA weights prioritize a barrier over a shuffle over a local
issue, while still allowing memory savings to offset communication costs.
They are deliberately weighted rather than a strict lexicographic ordering
of instruction counts. These are uncalibrated ranking heuristics, **not
cycle predictions or measured hardware latency ratios**. Shared-memory bank
conflicts, occupancy, instruction overlap, and expensive contribution
expressions are not modeled precisely. Keep the policy opt-in and benchmark
representative workloads before relying on its ranking.

## Diagnostics and validation

Enable `tl.enable_reducer_plan_verbose` to print candidate limits, spill,
execution and register costs, update widths, and collective statistics.
For a CUDA PrimFunc after `LayoutInference`, the read-only diagnostic
`tvm.get_global_func("tl.analysis.ReducerCost")(func)` returns per-reducer
plan and issue summaries, including actual wide fallback decisions.
The regression suite also compares predicted update widths against
read-modify-write accesses in the final device TIR.

```bash
python -m pytest testing/python/transform/test_tilelang_transform_reduction_aware_layout.py
python -m pytest testing/python/transform/test_tilelang_transform_reducer_scalar_candidates.py
python -m pytest testing/python/language/test_tilelang_language_reducer_v2.py
python maint/layout_inference/run.py --cute
python maint/layout_inference/run.py --anchor
```

Performance comparisons should include native, scalar, and intermediate-width
controls, full reductions, batch finalization, and representative softmax,
normalization, and GEMV kernels. Report compilation time as well as latency.
