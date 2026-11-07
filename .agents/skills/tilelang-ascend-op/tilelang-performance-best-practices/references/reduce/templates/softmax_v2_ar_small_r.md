# Softmax with a Small Reduction Axis

## Selection Criteria

Use this design when R is small enough for one task to batch multiple rows and improve vector utilization.

## TileLang/PTO Implementation

Use a fragment of shape `blk_m×padded_r`, with `T.reduce_max`/`T.reduce_sum` along the last dimension. Select `blk_m`, `threads`, and `padded_r` statically in the factory.

See `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py` for base reduction code: `T.copy` moves GM data into UB; `T.SimtVF` uses fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer`; tasks are assigned with a one-dimensional `T.Kernel`, and each output has only one owner.

## Correctness Gates

When the final batch contains fewer than `blk_m` rows, write only valid rows; padding rows must not access GM.

Keep max, sum, sums of squares, variance, rsqrt, exp, and online state in fp32 for low-precision inputs. Fill max tail lanes with negative infinity and sum tail lanes with zero. Cover tile±1, an extremely long reduction axis, cancellation, extreme values, the NaN/Inf contract, and the public forward/backward scope.

## Performance Gates

Search `blk_m` and `threads`, focusing measurements on launch overhead and coalescing of small copies.

After every targeted PTO correctness test passes, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and stage count. A multi-kernel design must include workspace and every launch; do not report only a local kernel.
