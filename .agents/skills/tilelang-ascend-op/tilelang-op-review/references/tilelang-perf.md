# TileLang Ascend High-Performance Programming Guidelines

<applicability>
Language: Python, TileLang DSL
Side: All
Domain: true
Triggers: T.copy, T.Persistent, T.Pipelined, T.annotate_buffer_versions, T.SimdVF, T.SimtVF, T.gemm, T.reduce_, alloc_shared, benchmark
Enabled by default: true
</applicability>

<review_load>
General review subagent rule capacity limit: 3
</review_load>

## Purpose

Review TileLang operators for performance, resource-usage, and numerical-accuracy issues.

## Quick Index

### Performance Guidelines (PERF-*)

| Rule ID | Rule name | Severity |
|---------|-----------|----------|
| PERF-1 | Avoid element-wise GM operations in hot loops | High |
| PERF-2 | Do not hard-code queryable hardware parameters | High |
| PERF-3 | Match multistage pipelines with buffer versioning | High |
| PERF-4 | Use reasonable transfer sizes and counts | Medium |
| PERF-5 | Avoid repeated GM reads | Medium |
| PERF-6 | Handle tail tiles correctly | High |

### Accuracy Guidelines (PREC-*)

| Rule ID | Rule name | Severity |
|---------|-----------|----------|
| PREC-1 | Preserve correct pipeline data dependencies | High |
| PREC-2 | Guard against division by zero | High |
| PREC-3 | Use sufficient intermediate accumulation precision for low-precision inputs | High |
| PREC-4 | Make special-value and numerical-stability behavior conform to the interface contract | High |

### Tiling Design Guidelines (TIL-*)

| Rule ID | Rule name | Severity |
|---------|-----------|----------|
| TIL-1 | Balance load across cores | Medium |
| TIL-2 | Do not exceed on-chip cache capacity | High |
| TIL-3 | Plan buffers appropriately | Medium |

## Applicable Scenarios

Ordinary Python tiling/dispatch code and TileLang kernels.

---

## Review Prerequisites

When reviewing version-dependent APIs, first locate the TileLang installation actually being imported, then inspect the current Ascend examples/tests and the selected Ascend backend lowering.
Capacity, alignment, and hardware-resource conclusions must be tied to the target SoC, the actual TileLang/PTO lowering, or a compilation resource report.

---

## Performance Guidelines

### PERF-1: Avoid Element-Wise GM Operations in Hot Loops

**Severity**: High

### Issue Description

Element-wise GM access or numerous small transfers in a hot loop reduce bandwidth utilization. For contiguous regions, prefer a GM↔UB `T.copy` followed by batched computation. Gather/scatter operations, short records, and complex branches require analysis of the actual access pattern; do not mechanically require every operation to use DMA.

### Incorrect Example

```python
for i in T.serial(valid):
    out[offset + i] = inp[offset + i]
```

### Correct Example

```python
T.copy(inp[offset : offset + valid], copy_ub[:valid])
T.copy(copy_ub[:valid], out[offset : offset + valid])
```

### Review Method

Locate the innermost hot loops and count GM loads/stores and `T.copy` operations. Report a performance issue only when the accesses are provably contiguous, can be safely combined, and occur on a hot path.

---

### PERF-2: Do Not Hard-Code Queryable Hardware Parameters

**Severity**: High

### Issue Description

For Vector, Cube, and mixed kernels, query the hardware to determine the available AIV, AIC, and paired resources, then allocate them according to the actual task count. Do not rely on project-private core-count query functions. Hard-coding the core count directly harms portability across devices.

### Incorrect Example

```python
num_cores = 20
```

### Correct Example

```python
num_cores = confirmed_available_aiv_core_count()
```

For a small workload, the core count generally should not exceed the number of independent tasks. If the core count is increased for scheduling within a specific L2 domain, document the reason.

---

### PERF-3: Match Multistage Pipelines with Buffer Versioning

**Severity**: High

### Issue Description

When `num_stages > 1`, producer/consumer buffers need enough versions, and the data dependencies in the first and final iterations must remain valid. The current Ascend AutoSchedule can select the version count automatically. An explicit `T.annotate_buffer_versions` overrides the scheduler's choice; it is not a mandatory ritual for enabling pipelining. Resident read-only data generally remains single-versioned.

### Correct Example

```python
T.annotate_buffer_versions({tile_ub: num_stages})
for w in T.Pipelined(num_outer_iters, num_stages=num_stages):
    ...
```

### Review Method

Inspect the writes, reads, stores, version count, and synchronization of every buffer in the loop body. When no annotation exists, inspect the actual version choice in the lowering/compilation result. Do not determine whether overlap took effect solely from source API names, and do not report an error merely because an annotation is absent. Additional versions must also be included in on-chip capacity calculations.

---

### PERF-4: Use Reasonable Transfer Sizes and Counts

**Severity**: Medium

### Issue Description

Small transfers increase instruction overhead, while an oversized tile may overflow UB, reduce parallelism, or waste work on tail tiles. Select tiles based on the contiguous dimension, dtype, 32B DMA granularity, and task count.

### Review Method

Calculate the byte count and loop count for each `T.copy`, determine whether adjacent contiguous copies can be combined, and check UB capacity and tail handling after combining them.
Without hot-path profiling or a provable order-of-magnitude regression, report only an optimization candidate, not `FAIL`.

---

### PERF-5: Avoid Repeated GM Reads

**Severity**: Medium

### Issue Description

If loop-invariant broadcast parameters, scales, or index tables are repeatedly transferred from GM in every task, they add deterministic GM traffic. However, keeping them resident also consumes UB/L1 capacity and may reduce the tile size or stage count.

### Review Method

1. Count reads of the same data per GM address and task loop.
2. Distinguish data invariant across tasks, data that differs by tile, and data intentionally reread because of a cache hint.
3. Calculate the resulting capacity, version count, and available parallelism before deciding whether reuse is worthwhile.

### Decision Method

Report the issue when static analysis proves that the same data is transferred redundantly with no compensating capacity benefit. Otherwise, list the reuse candidate and the representative cases that require validation.

---

### PERF-6: Handle Tail Tiles Correctly

**Severity**: High

### Issue Description

Tail tiles affect the valid GM range, UB padding, SIMD masks, task count, and output writeback simultaneously. Applying a mask only during computation cannot repair an out-of-bounds GM transfer that has already occurred. Shortening only the GM copy does not ensure that full-width register accesses remain within UB.

### Review Method

Check 0, 1, values immediately below and above alignment boundaries, values immediately below and above a tile boundary, and the largest public shape. Prove step by step that `offset`, `valid`, the source/destination slices, the UB footprint, and the writeback range agree. If the interface explicitly supports only certain remainder classes, verify that the wrapper/dispatch actually restricts the input accordingly.

### Decision Method

Assign `FAIL` when a valid input causes omitted or duplicate computation, an out-of-bounds access, or a read of uninitialized padding. If the interface does not promise general tail handling and the entry point already rejects unsupported shapes, do not require a new fallback.

---

## Accuracy Guidelines

### PREC-1: Preserve Correct Pipeline Data Dependencies

**Severity**: High

### Issue Description

Missing producer/consumer dependencies among DMA, SimdVF/SimtVF/Cube computation, and GM stores can read an old version, overwrite data still in use, or use an uninitialized version in the first iteration.

### Review Method

Build a read/write graph for every buffer in load→compute→store order. Check loop-carried dependencies, first and final iterations, consumption across execution domains, and manual multi-buffer indices. An explicit barrier must correspond to a specific, explainable hazard; do not use coarse-grained synchronization to hide an address or versioning error.

### Evidence Requirements

Report a conflict directly when the source proves it statically. When Ascend AutoSchedule provides automatic synchronization, evidence must include the current PTO lowering/generated code or a reproducible device error.

### PREC-2: Guard Against Division by Zero

**Severity**: High

Before a shape, group size, reduction count, norm denominator, or scale is used as a divisor, compile-time constraints, wrapper validation, or a kernel guard must guarantee that it is nonzero. Trace the divisor according to the divisor-source table in `tilelang-red-line.md`; `T.ceildiv` itself is not a guard.

### PREC-3: Use Sufficient Intermediate Accumulation Precision for Low-Precision Inputs

**Severity**: High

### Issue Description

Accumulating fp16/bf16/FP8/FP4 inputs directly at low precision during long reductions, variance calculations, softmax state updates, or GEMM can produce errors that grow with the reduction length, as well as overflow or underflow.

### Review Method

Starting from the input load, trace `T.cast`/SIMD `vcvt`, fragment/UB dtypes, reducer dtypes, L0C dtypes, and the final store. For fp16/bf16 reductions, norm, softmax statistics, and GEMM, use fp32 accumulation by default and convert back to the target dtype only at the boundary.

### Exclusion Rules

Low-precision accumulation is acceptable when the algorithm explicitly requires it or when the current implementation has error evidence covering the maximum reduction length and extreme-value inputs. Relaxing tolerance or reducing reference precision is not valid exclusion evidence.

---

### PREC-4: Make Special-Value and Numerical-Stability Behavior Conform to the Interface Contract

**Severity**: High

### Issue Description

NaN, Inf, positive and negative zero, all-zero rows, extremely large or small values, and cancellation inputs can change comparison, reduction, division, and type-conversion results. Stable algorithms such as softmax, norm, and scaling must preserve their mathematical transformations and interface conventions.

### Review Method

First determine special-value semantics from the public reference and tests. Then inspect the order of max/sum, `exp`/`log`, `rsqrt`, clamp, rounding, and saturation operations. Check lower-bound guards for all-zero denominators/amax values, and determine whether tail-padding values participate in valid reductions.

### Decision Method

Assign `FAIL` when the implementation clearly differs from the public reference's special-value semantics, or when a constructible input violates the output contract. If the interface does not define the semantics, mark the issue for confirmation and recommend adding a test.

---

## Tiling Design Guidelines

### TIL-1: Balance Load Across Cores

### Issue Description

Incorrect task-domain, wave-size, or core-ID mapping causes omissions or duplicates. Even when the mapping is correct, an extreme long tail, repeated fixed overhead on every core, or fewer tasks than cores may limit performance.

### Review Method

Expand the multidimensional task space of `T.Persistent(domain, num_cores, core_id)` and prove that every logical task is processed exactly once. Check cases with fewer tasks than cores, non-divisible counts, very small and very large shapes, imbalance across axes, and manual flattening/unflattening. Validate performance-related core-count choices on representative cases; report correctness mapping errors directly.

### TIL-2: Do Not Exceed On-Chip Cache Capacity

### Issue Description

A single shape written in source code does not represent actual resource usage. Buffer versions, alignment padding, resident data, temporary fragments, and registers after unrolling jointly determine capacity and spills.

### Review Method

For each buffer, calculate `element count × dtype.bytes × actual versions`, then add padding, resident data, and a safety margin. Check UB, L1, L0A/L0B/L0C separately. Determine register pressure from the current compilation result or generated code.

### Evidence Requirements

Capacity limits must come from the target SoC, an actual compilation resource report, or precise current-backend configuration. When the target capacity cannot be confirmed, mark the issue for confirmation and state the required compilation/resource evidence; do not apply a fixed capacity from another SoC to assign `FAIL`.

### TIL-3: Plan Buffers Appropriately

### Issue Description

Storing the same data more than once, adding versions with no benefit, extending lifetimes unnecessarily, or padding far beyond the access footprint consumes on-chip resources. Conversely, over-reusing one buffer may create opaque aliases and pipeline dependencies.

### Review Method

For each buffer, record its purpose, scope, shape, dtype, version count, first write, last read, and whether it remains resident across tasks. Recommendations to merge or reuse buffers must consider data dependencies, readability, capacity, and performance together; do not mechanically assume that fewer buffers are always better.

---

## Review Checklist

### Performance

- [ ] Hot-path GM accesses are batched appropriately
- [ ] Core counts come from the correct hardware query interface
- [ ] Pipelines and buffer versions are consistent
- [ ] Transfer volume, transfer count, and data reuse are appropriate
- [ ] Tail tiles are correct and do not cause abnormal degradation

### Accuracy

- [ ] Pipeline data dependencies and initialization are correct
- [ ] Every dynamic divisor has a nonzero guarantee
- [ ] Low-precision inputs use sufficiently precise intermediate state
- [ ] Special-value and stable-algorithm semantics match the public interface

### Tiling Design

- [ ] Task mapping is complete, duplicate-free, and load-balanced
- [ ] UB/L1/L0/register capacity is not exceeded
