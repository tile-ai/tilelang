# PTO Reduction, Norm, and Softmax

## Strategy Selection

Select full-load, recompute, online, or two-stage split-axis based on whether the reduction axis can remain resident in UB, whether there are enough output rows to saturate the vector cores, and whether every element must be emitted.

## TileLang/PTO Implementation

First implement an fp32 baseline that assigns one core to each output. Keep Norm weights resident; use stable max subtraction for softmax. Only when there are too few rows and the axis is extremely long should Kernel A write fp32 partial results and Kernel B merge them in a deterministic order.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: move GM data into UB with `T.copy`; use fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer` inside `T.SimtVF`; distribute tasks with a one-dimensional `T.Kernel`; and assign exactly one owner to each output.

When a short reduction fits in one SIMD register, an active-lane mask plus `S.vcadd`/`S.vdupv` can replace the SIMT reducer, while adjacent outputs are transferred into UB in batches. A/B testing must verify operation order, spills, and performance gains. If a partial result is consumed only by a downstream reduction, first evaluate merging the split axis within the producer and specializing `n_splits==1`, while preserving determinism and atomic semantics. When this structure applies, read [Batched Short Reduction](batched_short_reduction.md).

For normalization over a fixed small matrix, keep the current matrix and its backward gradient resident in SIMD registers throughout the entire computation, and write only the snapshots needed by the backward pass to UB. Compute one reciprocal per group before multiplying each element. Validate the local layout, synchronization, numerical error, register pressure, and unrolled code size. When this structure applies, read [Fixed Small State](fixed_small_state.md).

If each row first produces a reduction scalar and then uses that scalar to write the entire row element by element, read [Row-Wise Reduction with Element-Wise Writeback](rowwise_reduce_epilogue.md).

## Correctness Gate

For softmax, additionally check each row sum, repeated maxima, all-negative-infinity inputs, and positive-infinity semantics. For Norm, add `eps` to the fp32 statistic.

For low-precision inputs, keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32. Fill tail lanes with negative infinity for max and with 0 for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the documented forward/backward input domain.

## Performance Gate

Benchmark tile sizes, thread counts, resident weights, one-stage/two-stage variants, and split-axis in sequence; preserve a correct fallback.

When reduction computation is lightweight, the contiguous non-reduction axis is wide, and scheduling or DMA overhead for small tiles is significant, merge adjacent output elements as described in [Wide Output-Axis Tiling](wide_output_tiling.md).

After all targeted PTO correctness tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and pipeline stages. Multi-kernel approaches must account for the workspace and every launch; do not report only an individual kernel.
