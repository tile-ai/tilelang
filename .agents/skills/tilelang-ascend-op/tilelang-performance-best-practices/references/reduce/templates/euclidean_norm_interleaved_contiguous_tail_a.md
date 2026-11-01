# EuclideanNorm Contiguous Tail Tile

## Strategy Selection

Use this template when R is contiguous but the final tile is only partially filled.

## TileLang/PTO Implementation

Use `T.copy` to move valid elements into an fp32 tile, clear the remaining lanes to zero, and use a fragment to square the values and perform `reduce_sum`; keep the carry resident in fp32.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: move GM data into UB with `T.copy`; use fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer` inside `T.SimtVF`; distribute tasks with a one-dimensional `T.Kernel`; and assign exactly one owner to each output.

## Correctness Gate

Validate `R=1`, lane±1, tile±1, and the overflow boundary when squaring large values.

For low-precision inputs, keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32. Fill tail lanes with negative infinity for max and with 0 for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the documented forward/backward input domain.

## Performance Gate

Compare the latency of padded copy against explicit clearing.

After all targeted PTO correctness tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and pipeline stages. Multi-kernel approaches must account for the workspace and every launch; do not report only an individual kernel.
