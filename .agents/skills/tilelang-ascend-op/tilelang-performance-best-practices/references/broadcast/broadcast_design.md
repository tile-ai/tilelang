# Broadcast Implementation Overview

## Goals and Applicability

Use for elementwise operators with scalar, single-axis, or multi-axis broadcasting. Flatten the output space into independent tiles and compute input indexes from output indexes.

## TileLang/PTO Implementation

For contiguous broadcast axes, prefer moving a small input into UB once and reusing it in `T.SimdVF`; use a SIMD broadcast load for scalars. Use `T.SimtVF` for complex noncontiguous indexing, and specialize rank, axis, and stride through a Python factory. Assign output tasks with `T.Persistent`.

The implementation must use a one-dimensional `T.Kernel`. For vector-only tasks, derive the core count from the confirmed number of available AIV cores and use `min(core_count, independent_task_count)` to limit idle cores. Use `T.copy` between GM and UB/L1; use `T.SimdVF` and `T.Parallel` for contiguous regular computation; use `T.Persistent` or `T.Pipelined` for tasks spanning tiles; and declare multiversioned buffers explicitly with `T.annotate_buffer_versions`.

## Accuracy Gates

The dtype-conversion order before and after broadcasting must match the reference. Promote low-precision transcendental functions and accumulation to fp32. Validate size-1 axes, multiple broadcast axes, zero-dimensional scalars, and tail blocks.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, non-32B tails, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Passing by relaxing tolerances, reducing reference precision, or skipping cases is prohibited.

## Performance Gates

Measure repeated GM reads, indexing overhead, and vector utilization in particular. Keep small broadcast inputs resident in UB to avoid repeated movement for every output tile.

First run targeted accuracy tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat, device, and concurrency settings. Report kernel latency, effective GM bandwidth, UB occupancy, stage count, and differences from baseline.

## Executable Code and Evidence

Reuse the pipeline skeleton from `examples/ascend/example_simdvf_vecadd.py` and the indexing pattern from `examples/ascend/example_simdvf_per_token_cast_to_fp8.py`.
