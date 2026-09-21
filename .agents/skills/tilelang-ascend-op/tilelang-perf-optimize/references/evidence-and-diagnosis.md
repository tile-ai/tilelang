# Evidence and Bottleneck Attribution

## Priority of Facts

1. Tests, public interfaces, dispatch, and the target `_asc.py` in the current repository.
2. Current-round PTO compilation results, per-case kernel latency, profiling reports, and traces.
3. Actually imported TileLang source, tests, and `examples/ascend/`.
4. Maturity-rated references in the Skill named `tilelang-performance-best-practices`.

Lower-priority material cannot override higher-priority measured results. Data from another version, device, or measurement methodology can form only a hypothesis that still requires validation.

## Collection Checklist

Record at least the following for every case:

| Field | Requirement |
|---|---|
| Case ID / shape / dtype / attrs | Correspond one-to-one with the tests and input list |
| Dispatch branch | Derive from and verify against actual code conditions |
| Kernel identity | Confirm that profiling targets the intended compiled artifact |
| Latency | Use the same device, warmup, repeat, concurrency, and timing layer |
| Correctness | The baseline must satisfy the thresholds established by the test file |
| Profiling metrics | Record only fields and units actually provided by the tool |
| Trace observations | Mark the interval in which each occurs; do not substitute one screenshot for all cases |

## Bottleneck Classification

Classification interprets data; it is not a lookup table of fixed thresholds.

| Type | Evidence to Observe | Common Hypotheses Requiring Validation |
|---|---|---|
| Compute-bound | Compute execution dominates, with no clear improvement after reducing transfers | Reduce operations, fuse expressions, or change reduction/GEMM dataflow |
| Transfer-bound | Read/write time or bandwidth dominates, and compute units wait for data | Increase contiguous transfers, improve reuse, or reduce GM round trips |
| Scalar/scheduling-bound | Small tasks, loops, and address preparation take a large share, leaving compute units underutilized | Merge small tiles, hoist invariants, or reduce branches or loop nesting |
| Parallelism-bound | Independent tasks are fewer than available cores, or per-core work has a large tail imbalance | Change the partition axis, adjust core count, or split or merge tasks |
| Launch-bound | The kernel body is very short, and fixed overhead dominates end-to-end time | Fuse kernels or combine dispatch; compare both kernel and end-to-end measurements |
| Mixed bottleneck | Multiple signals are similar or change with shape | Handle per-case branches and validate only one primary variable at a time |

Do not infer bound type solely from the operator category, and do not use one case's metrics to represent every case. When metric ratios are close, prioritize small controlled experiments, such as changing only tile size, core count, or buffer versions, and observe whether latency changes according to the hypothesis.

## Conclusion Strength

- **Measured conclusion**: Directly supported by current-round data and eligible for solution ranking.
- **Hypothesis requiring validation**: An observation exists, but an isolated experiment is missing; it may form an experimental solution.
- **Unknown**: Required data is missing; return to the collection stage.

Every optimization measure in the report should be traceable to one of these conclusion-strength categories.
