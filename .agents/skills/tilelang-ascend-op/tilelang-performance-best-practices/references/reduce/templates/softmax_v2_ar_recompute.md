# Softmax AR Recompute

## Selection

Use this design when the entire row cannot remain resident in UB, every element must be output, and additional reads are acceptable in exchange for lower capacity requirements.

## TileLang/PTO Implementation

In the first pass, compute the global max tile by tile. In the second pass, compute sum(exp(x-max)). In the third pass, reread the input and produce the output. Reuse one fp32 tile in each pass, while keeping max/sum state resident.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: transfer GM data into UB with T.copy; in T.SimtVF, use an fp32 fragment with T.reduce_max/T.reduce_sum or alloc_reducer/finalize_reducer. Assign tasks with a one-dimensional T.Kernel, ensuring that each output has exactly one owner.

## Accuracy Gates

Use consistent tail identity values across all three passes; the exp expressions in the second and third passes must be identical.

Keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32 for low-precision inputs. Fill tail lanes with negative infinity for max and zero for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the public forward/backward ranges.

## Performance Gates

Compare GM reads and exp costs against the two-pass online-statistics design; retain recompute as a robust fallback.

After all targeted PTO accuracy tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and stage. Multi-kernel designs must account for workspace and every launch; do not report only a local kernel.
