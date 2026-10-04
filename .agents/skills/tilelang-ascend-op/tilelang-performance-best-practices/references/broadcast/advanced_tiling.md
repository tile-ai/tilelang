# Specialized Tiling for Broadcast Shapes

## Goals and Applicability

Use this approach for operators whose behavior differs significantly across ranks, broadcast axes, and broadcast factors.

## TileLang/PTO Implementation

In the Python factory, select the tile, core count, SimdVF/SimtVF execution domain, and stage count according to contiguity, inner length, broadcast factor, and dtype. Retain a correct fallback covering every valid shape.

The implementation must use a one-dimensional T.Kernel. For pure Vector tasks, obtain the core count from confirmed_available_aiv_core_count() and use min(core_count, independent_task_count) to avoid idle cores. Use T.copy between GM and UB/L1. Use T.SimdVF and T.Parallel for contiguous regular computation. Use T.Persistent or T.Pipelined for cross-tile tasks, and explicitly declare multi-version buffers with T.annotate_buffer_versions.

## Correctness Gate

Run correctness tests on both sides of every dispatch boundary. For dynamic shapes, T.assume may express only guarantees actually provided by the caller.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, tails not aligned to 32B, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Do not obtain a passing result by widening tolerances, reducing reference precision, or skipping cases.

## Performance Gate

Search tiles and stages systematically, changing only one variable at a time. Track the number of compiled variants and cache usage to avoid excessive specialization.

First run targeted correctness tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat count, device, and concurrency. Report kernel latency, effective GM bandwidth, UB usage, stage count, and the difference from the baseline.

## Executable Code and Evidence

For configuration selection, refer to `bf16_select_config` in the actual TileLang source file `examples/ascend/example_gemm_various_shapes.py`; verify the currently installed version before use.
