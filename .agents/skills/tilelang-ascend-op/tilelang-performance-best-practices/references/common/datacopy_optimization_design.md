# T.copy Transfers and Memory-Access Coalescing

## Goals and Applicability

Use this approach for transfers from GM to UB/L1/L0 and for writeback in the reverse direction. The goals are to reduce the number of transfers, increase contiguous burst sizes, and overlap transfers with computation.

## TileLang/PTO Implementation

Prefer copying contiguous rectangular slices. For tails, use an exact valid slice or the pad_value argument of T.copy. Move repeatedly read weights, scales, and lookup tables into a single-version buffer outside the Persistent loop. Use T.StridedTensor for strided inputs. Only l2_cache_ctrl values proven by target examples may enter the production path.

Coalesce adjacent short records or scalars along the outer dimension into rectangular/contiguous transfers to avoid a 4B GM access for every token. Also verify nburst, burst_len, invalid bytes, and UB usage.

The implementation must use a one-dimensional T.Kernel. For pure Vector tasks, obtain the core count from confirmed_available_aiv_core_count() and use min(core_count, independent_task_count) to avoid idle cores. Use T.copy between GM and UB/L1. Use T.SimdVF and T.Parallel for contiguous regular computation. Use T.Persistent or T.Pipelined for cross-tile tasks, and explicitly declare multi-version buffers with T.annotate_buffer_versions.

## Correctness Gate

Promote low-precision inputs to fp32 in UB according to the operator semantics. pad_value must be the identity element for the operation, such as negative infinity for max and 0 for sum. Never read uninitialized padding lanes.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, tails not aligned to 32B, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Do not obtain a passing result by widening tolerances, reducing reference precision, or skipping cases.

## Performance Gate

Compare total GM bytes, copy-instruction count, MTE2/MTE3 time, and effective bandwidth. Retain coalesced transfers only when they do not increase invalid data volume or UB pressure.

First run targeted correctness tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat count, device, and concurrency. Report kernel latency, effective GM bandwidth, UB usage, stage count, and the difference from the baseline.

## Executable Code and Evidence

Refer to examples/ascend/example_copy_pad_value.py, examples/ascend/example_simdvf_vecadd.py, and examples/ascend/example_gemm.py.
