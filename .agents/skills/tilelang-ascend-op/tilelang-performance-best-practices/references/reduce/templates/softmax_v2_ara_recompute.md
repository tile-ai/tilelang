# ARA Recompute Softmax

## Selection Criteria

Use this design when the middle axis is very long and the online implementation does not fit the current stride or lowering.

## TileLang/PTO Implementation

Use three passes for max, sum, and output, with every pass accessing data through the same outer/inner/R mapping. Specialize the stride category through the factory and retain a general SIMT fallback.

See `references/reduce/templates/dav310/kernel_utils.py` and `examples/ascend/example_rmsnorm.py` for base reduction code: `T.copy` moves GM data into UB; `T.SimtVF` uses fp32 fragments with `T.reduce_max`/`T.reduce_sum` or `alloc_reducer`/`finalize_reducer`; tasks are assigned with a one-dimensional `T.Kernel`, and each output has only one owner.

## Correctness Gates

Verify that the address mapping is exactly identical across all three passes.

Keep max, sum, sums of squares, variance, rsqrt, exp, and online state in fp32 for low-precision inputs. Fill max tail lanes with negative infinity and sum tail lanes with zero. Cover tile±1, an extremely long reduction axis, cancellation, extreme values, the NaN/Inf contract, and the public forward/backward scope.

## Performance Gates

Use this as a correct fallback and perform end-to-end A/B comparisons against online or fused-layout variants.

After every targeted PTO correctness test passes, compare end-to-end latency, effective GM bytes, Vector/MTE time, core utilization, UB bytes, and stage count. A multi-kernel design must include workspace and every launch; do not report only a local kernel.
