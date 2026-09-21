# Unaligned Tail-Block Implementation

## Goals and Applicability

Use for paths where the element count, row width, or matrix dimension is not divisible by the DMA, SIMD, or tile granularity.

## TileLang/PTO Implementation

Access only valid elements on the GM side. Allocate UB for the complete register footprint that may be touched. Write the identity element to invalid lanes in reductions; for transpose/gather, write back only the valid output rectangle. When a Python factory can generate specialized kernels by remainder class, prioritize eliminating dynamic branches from the hot loop.

The implementation must use a one-dimensional `T.Kernel`. For vector-only tasks, derive the core count from the confirmed number of available AIV cores and use `min(core_count, independent_task_count)` to limit idle cores. Use `T.copy` between GM and UB/L1; use `T.SimdVF` and `T.Parallel` for contiguous regular computation; use `T.Persistent` or `T.Pipelined` for tasks spanning tiles; and declare multiversioned buffers explicitly with `T.annotate_buffer_versions`.

## Accuracy Gates

For each dtype, validate DMA alignment, SIMD lane count, and the converted fp32 footprint separately. Use canaries to verify that no out-of-bounds writes occur before or after the tail.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, non-32B tails, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Passing by relaxing tolerances, reducing reference precision, or skipping cases is prohibited.

## Performance Gates

Benchmark full tiles and tile±1 separately to ensure that branches added for tails do not slow the main path. If too many specialized kernels create compilation or cache pressure, merge infrequent remainder classes.

First run targeted accuracy tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat, device, and concurrency settings. Report kernel latency, effective GM bandwidth, UB occupancy, stage count, and differences from baseline.

## Executable Code and Evidence

Refer to `testing/ascend/layout/test_ascend_l0_transpose.py` and `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`.
