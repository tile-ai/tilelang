# Scan Kernel Integration Workflow

## Applicability

Use this workflow when adding a cumulative operator to the current repository.

## TileLang/PTO Implementation

The Python wrapper handles empty tensors and selects either row-owned or split-row execution. The factory specializes `tile_cols`, dtype, and mode; enable dynamic rows/cols only when the backend has been validated for them.

For the correctness baseline, see `references/scan/templates/dav310/scan_base.py`: each row is owned by exactly one kernel task, the last contiguous axis is partitioned into tiles, the fp32 carry remains resident, and GM/UB transfers use `T.copy`. The implementation does not call `T.cumsum` or rely on a grid barrier.

## Correctness Gate

Declare inclusive/exclusive, forward/reverse, axis, and output dtype semantics first.

Cover `cols=1`, 63/64/65, 127/128/129, tile±1, multiple tiles, and the maximum length; test row counts below, equal to, and above the number of vector cores; and cover positive-negative cancellation, mixed magnitudes, NaN/Inf, and the overflow contract. Compare against `torch.cumsum(x.float(), dim=-1)` and check every carry across tile boundaries.

## Performance Gate

Ensure targeted correctness tests cover every dispatch path before running benchmarks with the same methodology.

First pass PTO lowering and targeted correctness tests. Compare the complete end-to-end latency of the serial-correctness baseline, SIMD lane scan, one-stage/two-stage DMA, and the three-kernel split-row approach. Report effective bandwidth, Vector/MTE time, core utilization, workspace bytes, and all launches.
