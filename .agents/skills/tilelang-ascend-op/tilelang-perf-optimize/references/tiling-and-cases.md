# Tiling and Case Modeling

## Reconstruct the Execution Path First

For every case, record the path downward from the public entry point: dispatch condition → build parameters → `@T.prim_func` → `T.Kernel` task mapping → tile loop → execution domain. Identical operator names do not imply identical execution paths; broadcast axes, reduction axes, dtype, alignment, and tail tiles can all change the selected branch.

The following table is recommended:

| Case | Semantic path | dtype/accumulation | dispatch | Core count | tile | Tail tile | Baseline bottleneck |
|---|---|---|---|---:|---|---|---|

Combine cases into one model only when both their execution paths and bottlenecks are similar.

## General Model

### Tasks and Core Count

- `independent_tasks`: The number of tasks that can run in parallel without cross-task dependencies.
- `used_cores <= independent_tasks`.
- `tasks_per_core = ceil(independent_tasks / used_cores)`.
- Record the workloads of both the most-loaded and least-loaded cores; an average does not reveal tail imbalance.

The core-count source and the vector/cube/mixed selection must be confirmed through the skill named `tilelang-performance-best-practices` and the current repository configuration.

### Tiles and Loops

- `tiles_per_task = ceil(valid_extent / tile_extent)`.
- For each tile, distinguish `valid_extent` from the `padded_extent` prepared for SIMD/DMA/layout requirements.
- Use the actual tile count when estimating fixed overhead, count only valid reads and writes when estimating GM bytes, and use the complete padded footprint when estimating UB usage.

For small shapes, increasing the tile size or combining tasks may reduce loop and scheduling overhead. For large shapes, an oversized tile may reduce parallelism or increase buffer pressure. Both effects must be validated with measurements.

### Buffer Budget

Calculate each buffer separately:

`bytes = product(padded_shape) * dtype_bytes * versions`

The total budget must also include resident data, temporary results, alignment padding, and a safety margin. Determine the version relationship between `T.annotate_buffer_versions` and `T.Persistent(..., num_stages=N)` from the current TileLang implementation and validated examples; do not assume it.

### Data Flow

For each tile, list:

- GM bytes read and written back.
- Lifetimes of intermediate results in UB/L1/L0.
- The number of repeated reads of the same data.
- Operation count and reduction/GEMM accumulation state.
- Whether load, computation, and writeback have verifiable opportunities for overlap.

## Parameter Output

For every parameter in a plan, provide the current value, candidate value, derivation, applicable cases, capacity/alignment gates, and rollback conditions. Example:

| Parameter | Current | Candidate | Rationale | Applicable cases | Gate |
|---|---:|---:|---|---|---|
| `tile_n` | 128 | 256 | Loop overhead dominates small cases | group-S | UB remains within budget after including versions and padding |
| `used_cores` | 20 | 8 | There are only 8 independent tasks | group-tail | Does not exceed the task count; no per-case latency regression |

A candidate value that has not been verified against the current-version API, lowering, and capacity may only be marked for validation; it must not be presented as a final configuration.
