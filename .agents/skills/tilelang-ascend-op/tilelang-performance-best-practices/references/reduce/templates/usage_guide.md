# Reduction Kernel Usage Workflow

## Selection

Use this workflow when adding ReduceSum/Max, LayerNorm, RMSNorm, or Softmax.

## TileLang/PTO Implementation

A Python factory accepts a static reduction width, dtype, threads, tile, and stages, and returns T.prim_func. Express dynamic rows with T.dynamic. The core count is min(num_cores, rows), where num_cores is passed explicitly from hardware-query results. Reuse an existing kernel before adding shape specialization.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: transfer GM data into UB with T.copy; in T.SimtVF, use an fp32 fragment with T.reduce_max/T.reduce_sum or alloc_reducer/finalize_reducer. Assign tasks with a one-dimensional T.Kernel, ensuring that each output has exactly one owner.

## Accuracy Gates

Submit the test file together with the kernel, and use torch fp32 or a higher-precision stable expression as the reference.

Keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32 for low-precision inputs. Fill tail lanes with negative infinity for max and zero for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the public forward/backward ranges.

## Performance Gates

After targeted accuracy tests, run the relevant full test suite and record every dispatch boundary.

After all targeted PTO accuracy tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and stage. Multi-kernel designs must account for workspace and every launch; do not report only a local kernel.
