# UB/L1 Residency and Multiversion Buffers

## Objective and Applicability

Use this design for weights, scales, lookup tables, and reduction state reused across multiple tiles.

## TileLang/PTO Implementation

Allocate resident data outside `T.Persistent`/`T.Pipelined`, transfer it only once, and do not include it in `T.annotate_buffer_versions`. Multiversion input and output tiles according to `num_stages`; keep reduction carry, online-softmax max/sum, and scan carry single-version. The budget is the sum, over all buffers, of element count multiplied by dtype bytes and version count, plus padding and a safety margin.

The implementation must use a one-dimensional `T.Kernel`. For a vector-only workload, use the confirmed available AIV core count and cap it with `min(core count, independent task count)` to avoid idle cores. Use `T.copy` between GM and UB/L1; use `T.SimdVF` and `T.Parallel` for regular contiguous computation; use `T.Persistent` or `T.Pipelined` for cross-tile tasks, and explicitly declare multiversion buffers with `T.annotate_buffer_versions`.

## Correctness Gates

Resident low-precision weights may retain their original dtype; statistical state and long-chain accumulation remain fp32. Confirm that no task reads residual state from a previous task.

Cover minimum, common, and maximum shapes; tile-1/tile/tile+1; non-32B tails; zero; positive and negative extremes; and interface-defined NaN/Inf semantics. Do not obtain a pass by expanding tolerances, lowering reference precision, or skipping cases.

## Performance Gates

A/B test repeated GM traffic, UB occupancy, occupancy, and pipeline overlap. When residency reduces the stage count or causes spills, compare total latency before deciding.

First run targeted correctness tests with `TILELANG_DEFAULT_TARGET=pto`, then run the relevant complete test suite. After everything passes, measure performance using identical inputs, dtype, warmup, repeat, device, and concurrency. Report kernel latency, effective GM bandwidth, UB occupancy, stage count, and the difference from baseline.

## Executable Code and Evidence

For references, see weight residency in `examples/ascend/example_rmsnorm.py` and fp32 online state in `examples/ascend/flash_attention/example_mha.py`.
