# EuclideanNorm Multi-Row Interleaving

## Selection

Use this design when R is small, allowing multiple rows to be processed at once with reduction performed within each row.

## TileLang/PTO Implementation

Use a blk_m×padded_r fragment and apply T.reduce_sum along R; write back only valid rows in the final batch.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: transfer GM data into UB with T.copy; in T.SimtVF, use an fp32 fragment with T.reduce_max/T.reduce_sum or alloc_reducer/finalize_reducer. Assign tasks with a one-dimensional T.Kernel, ensuring that each output has exactly one owner.

## Accuracy Gates

Ensure that accumulation never crosses row boundaries.

Keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32 for low-precision inputs. Fill tail lanes with negative infinity for max and zero for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the public forward/backward ranges.

## Performance Gates

Search blk_m/threads to fully utilize copy and reduction lanes.

After all targeted PTO accuracy tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and stage. Multi-kernel designs must account for workspace and every launch; do not report only a local kernel.
