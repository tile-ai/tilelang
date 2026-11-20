# Reduction Tiling for a Wide Output Axis

## Objective and Applicability

Use this approach for reductions with lightweight computation and a wide, contiguous non-reduction axis when small tiles cause excessive task, DMA, or address-preparation overhead. It optimizes partitioning of the output axis without changing the reduction axis or numerical operation order.

## TileLang/PTO Implementation

- Increase the tile size along the contiguous non-reduction axis so that one Persistent task processes multiple SIMD chunks consecutively. Search candidates in a geometric progression rather than hard-coding empirical values.
- Compute `tasks = outer * ceil(output_inner / tile_inner)`, valid and padded element counts, the number of DMA operations, per-core waves, and UB bytes for every versioned buffer together.
- Re-select the core count after increasing the tile size. Avoid allowing insufficient parallelism, tail-tile waste, UB overflow, or multi-stage pressure to offset the gains. A/B test tile size and stage count independently.
- Use a Python factory to dispatch by static shape, dtype, and reduction size, while preserving a correct fallback.

## Validation Gate

Preserve fp32 state for low-precision reductions and a single owner for each output. Cover reduction sizes, tile±1, partial tail tiles, extremely small and extremely large output axes, and forward/backward; GM accesses must be limited to valid elements. Compare task and DMA counts, average transfer granularity, Scalar/MTE/Vector time, core utilization, UB usage, and latency measured with the same methodology.
