# Two-Stage Split-R for EuclideanNorm

## Selection Criteria

Use this design when there are very few output tasks and R is extremely long, so a single owner cannot occupy all vector cores.

## TileLang/PTO Implementation

Kernel A computes fp32 partial sums of squares over each row × R-partition and writes them to workspace. Kernel B assigns one unique owner per row to perform a deterministic reduction followed by sqrt. The host wrapper launches them sequentially.

See `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py` for base reduction code: `T.copy` moves GM data into UB; `T.SimtVF` uses fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer`; tasks are assigned with a one-dimensional `T.Kernel`, and each output has only one owner.

## Correctness Gates

Keep partials, merge state, and the state before sqrt in fp32. Cover numerical error across different partition counts.

Keep max, sum, sums of squares, variance, rsqrt, exp, and online state in fp32 for low-precision inputs. Fill max tail lanes with negative infinity and sum tail lanes with zero. Cover tile±1, an extremely long reduction axis, cancellation, extreme values, the NaN/Inf contract, and the public forward/backward scope.

## Performance Gates

Include the two launches and workspace traffic; enable this design only when measurements show it is faster for extremely long R.

After every targeted PTO correctness test passes, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and stage count. A multi-kernel design must include workspace and every launch; do not report only a local kernel.
