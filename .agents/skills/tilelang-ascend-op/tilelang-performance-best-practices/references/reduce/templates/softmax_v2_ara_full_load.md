# ARA Full-load Softmax

## Strategy Selection

Use this template when the reduction axis is an intermediate dimension, outer×inner produces enough independent outputs, and the corresponding slice can remain resident.

## TileLang/PTO Implementation

Treat each outer/inner pair as one output and gather along R into an fp32 fragment. If the source strides can be coalesced into a rectangle, use `T.copy` first; otherwise, use validated SIMT indexing.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: move GM data into UB with `T.copy`; use fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer` inside `T.SimtVF`; distribute tasks with a one-dimensional `T.Kernel`; and assign exactly one owner to each output.

## Correctness Gate

Validate normalization along every supported axis and noncontiguous strides.

For low-precision inputs, keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32. Fill tail lanes with negative infinity for max and with 0 for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the documented forward/backward input domain.

## Performance Gate

Compare the total GM bytes and latency of contiguous softmax after transposition against direct strided access.

After all targeted PTO correctness tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and pipeline stages. Multi-kernel approaches must account for the workspace and every launch; do not report only an individual kernel.
