# Resident Reduction State

## Strategy Selection

Use this pattern when only a small amount of state must persist across input tiles, such as sum, max, sum of squares, or the `m`/`l` values of online softmax.

## TileLang/PTO Implementation

Allocate the state as a single-version fp32 UB buffer or fragment. Initialize it before the tile loop, update it inside the loop, and write it back after the loop. Input and output tiles may use multiple buffer versions, but the state must not be included in `buffer_versions`.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: move GM data into UB with `T.copy`; use fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer` inside `T.SimtVF`; distribute tasks with a one-dimensional `T.Kernel`; and assign exactly one owner to each output.

## Correctness Gate

Reinitialize the state for every new output to prevent contamination across tasks.

For low-precision inputs, keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32. Fill tail lanes with negative infinity for max and with 0 for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the documented forward/backward input domain.

## Performance Gate

Compare the reduction in GM partial-result traffic against the added serial dependency. When the state is small but the dependency chain is long, focus optimization on DMA overlap.

After all targeted PTO correctness tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and pipeline stages. Multi-kernel approaches must account for the workspace and every launch; do not report only an individual kernel.
