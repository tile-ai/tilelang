# Multidimensional Broadcast and Rectangular Data Movement

## Goals and Applicability

Use for multi-axis broadcasts that can be collapsed into contiguous rectangles.

## TileLang/PTO Implementation

First merge adjacent contiguous dimensions in the host/factory to produce outer, broadcast, and inner segments. `T.copy` handles only contiguous rectangles; explicitly index noncontiguous dimensions that it cannot express in `T.SimtVF`. Do not assume that an arbitrary N-dimensional broadcast can be lowered to a single DMA operation.

The implementation must use a one-dimensional `T.Kernel`. For vector-only tasks, derive the core count from the confirmed number of available AIV cores and use `min(core_count, independent_task_count)` to limit idle cores. Use `T.copy` between GM and UB/L1; use `T.SimdVF` and `T.Parallel` for contiguous regular computation; use `T.Persistent` or `T.Pipelined` for tasks spanning tiles; and declare multiversioned buffers explicitly with `T.annotate_buffer_versions`.

## Accuracy Gates

Validate logical shapes, strides, and storage offsets before and after merging. Use `T.StridedTensor` for noncontiguous views.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, non-32B tails, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Passing by relaxing tolerances, reducing reference precision, or skipping cases is prohibited.

## Performance Gates

Compare the copy count, scalar address calculations, and total GM bytes before and after dimension merging. Use the DMA path only when generated code confirms that rectangular movement is realized.

First run targeted accuracy tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat, device, and concurrency settings. Report kernel latency, effective GM bandwidth, UB occupancy, stage count, and differences from baseline.

## Executable Code and Evidence

For dynamic dimensions and tensor views, refer respectively to `testing/ascend/language/test_tilelang_ascend_dynamic_none.py` and `tilelang/language/symbolics.py`. Verify movement constraints against `src/ascend/op/copy.cc`.
