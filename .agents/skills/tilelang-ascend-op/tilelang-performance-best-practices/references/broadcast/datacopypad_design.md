# Broadcast Tail Padding

## Goals and Applicability

Use this approach when the final block of a broadcast input or output does not satisfy alignment requirements.

## TileLang/PTO Implementation

Use T.copy(..., pad_value=value) only for shapes supported by the target PTO examples. Pad max/topk operations with negative infinity and sum operations with 0. Always write only the valid output region. Distinguish the data padding value from the final output value.

The implementation must use a one-dimensional T.Kernel. For pure Vector tasks, obtain the core count from confirmed_available_aiv_core_count() and use min(core_count, independent_task_count) to avoid idle cores. Use T.copy between GM and UB/L1. Use T.SimdVF and T.Parallel for contiguous regular computation. Use T.Persistent or T.Pipelined for cross-tile tasks, and explicitly declare multi-version buffers with T.annotate_buffer_versions.

## Correctness Gate

Cover final blocks with 1, lane-1, and lane+1 valid elements, and verify negative infinity, NaN, and dtype conversion behavior.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, tails not aligned to 32B, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Do not obtain a passing result by widening tolerances, reducing reference precision, or skipping cases.

## Performance Gate

A/B test padded copy against explicit fill+copy, and record additional UB writes and copy latency.

First run targeted correctness tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat count, device, and concurrency. Report kernel latency, effective GM bandwidth, UB usage, stage count, and the difference from the baseline.

## Executable Code and Evidence

Refer directly to examples/ascend/example_copy_pad_value.py and examples/ascend/example_simdvf_topk_gate.py.
