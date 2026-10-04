# UB-Resident Broadcast

## Goals and Applicability

Use when the same small tensor is read repeatedly by multiple output tiles or across multiple broadcast factors.

## TileLang/PTO Implementation

Outside each core's task loop, use `T.copy` to move the small tensor into single-version UB. Inside the loop, load only the output tile and read from the resident buffer. If the small tensor is shared across cores, have each core load its own copy rather than relying on unverified cross-core sharing.

The implementation must use a one-dimensional `T.Kernel`. For vector-only tasks, derive the core count from the confirmed number of available AIV cores and use `min(core_count, independent_task_count)` to limit idle cores. Use `T.copy` between GM and UB/L1; use `T.SimdVF` and `T.Parallel` for contiguous regular computation; use `T.Persistent` or `T.Pipelined` for tasks spanning tiles; and declare multiversioned buffers explicitly with `T.annotate_buffer_versions`.

## Accuracy Gates

Resident data on every core must be initialized completely. Dynamic-length tail padding must not contribute to valid output.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, non-32B tails, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Passing by relaxing tolerances, reducing reference precision, or skipping cases is prohibited.

## Performance Gates

Calculate the per-core copy cost and reuse count for resident data. Enable residency only when the saved GM traffic exceeds the initialization cost and UB usage does not constrain the pipeline.

First run targeted accuracy tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat, device, and concurrency settings. Report kernel latency, effective GM bandwidth, UB occupancy, stage count, and differences from baseline.

## Executable Code and Evidence

For the weight-residency pattern, refer to `examples/ascend/example_rmsnorm.py`.
