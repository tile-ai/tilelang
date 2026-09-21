# Softmax AR Full-load

## Strategy Selection

Use this template when the entire input row, the fp32 workspace, and all required buffers fit in UB simultaneously.

## TileLang/PTO Implementation

Use one `T.copy` to move the entire row into an fp32 UB buffer or fragment, perform `reduce_max`, compute `exp(x-max)`, perform `reduce_sum`, normalize, and write the result back. Set alignment-padding elements to the reduction identity before they participate in the reduction.

For baseline reduction code, see `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py`: move GM data into UB with `T.copy`; use fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer` inside `T.SimtVF`; distribute tasks with a one-dimensional `T.Kernel`; and assign exactly one owner to each output.

## Correctness Gate

Compare the numerically stable form against an fp32 reference, and convert to the output dtype only at the final store.

For low-precision inputs, keep max, sum, sum of squares, variance, rsqrt, exp, and online state in fp32. Fill tail lanes with negative infinity for max and with 0 for sum. Cover tile±1, extremely long reduction axes, cancellation, extreme values, the NaN/Inf contract, and the documented forward/backward input domain.

## Performance Gate

For small and medium R, compare single-stage and two-stage row pipelines; avoid forcing multiple variants within a single row.

After all targeted PTO correctness tests pass, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and pipeline stages. Multi-kernel approaches must account for the workspace and every launch; do not report only an individual kernel.
