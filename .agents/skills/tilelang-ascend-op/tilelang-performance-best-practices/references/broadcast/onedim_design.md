# Single-Axis Broadcast

## Goals and Applicability

Use this approach when one input axis has length 1, or when a contiguous data segment is replicated along an adjacent dimension.

## TileLang/PTO Implementation

Decompose the linear output index into outer, broadcast index, and inner components. When inner is contiguous, each task transfers the source slice to UB once and writes multiple destination positions in a SIMD loop. For a scalar inner component, use a broadcast register.

The implementation must use a one-dimensional T.Kernel. For pure Vector tasks, obtain the core count from confirmed_available_aiv_core_count() and use min(core_count, independent_task_count) to avoid idle cores. Use T.copy between GM and UB/L1. Use T.SimdVF and T.Parallel for contiguous regular computation. Use T.Persistent or T.Pipelined for cross-tile tasks, and explicitly declare multi-version buffers with T.annotate_buffer_versions.

## Correctness Gate

Cover broadcast sizes of 1 and 2, vector lane±1, and large broadcast factors. Verify that index multiplication uses a sufficiently wide integer type.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, tails not aligned to 32B, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Do not obtain a passing result by widening tolerances, reducing reference precision, or skipping cases.

## Performance Gate

For a large broadcast factor, measure whether source GM bytes approach a single read. For a small factor, ensure that the fixed overhead of UB staging does not exceed direct-access overhead.

First run targeted correctness tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat count, device, and concurrency. Report kernel latency, effective GM bandwidth, UB usage, stage count, and the difference from the baseline.

## Executable Code and Evidence

Base the implementation on examples/ascend/example_simdvf_vecadd.py. Verify dynamic-stride behavior against the actual interfaces in `tilelang/language/symbolics.py` and `src/ascend/op/copy.cc`.
