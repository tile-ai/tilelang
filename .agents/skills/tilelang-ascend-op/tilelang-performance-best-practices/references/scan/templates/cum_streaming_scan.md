# Streaming Scan

## Applicability

Use this approach when there are enough rows, `R` can be very large, and the sequential dependency within each row cannot be eliminated.

## TileLang/PTO Implementation

The owner of each row maintains an fp32 running state. For each tile, load the data, compute the local prefix, add the carry, write the result back, and update the last valid value. Keep the carry single-version; a two-stage pipeline may be attempted for input and output.

The correctness baseline is `references/scan/templates/dav310/scan_base.py`: each row is owned by exactly one kernel task, the final contiguous axis is tiled, the fp32 carry remains resident, and GM/UB transfers use `T.copy`. The implementation does not call `T.cumsum` or depend on a grid barrier.

## Correctness Gate

For extrema that include an index, update the value and index together, and explicitly define whether equal values select the first or last occurrence.

Cover `cols=1`, `63/64/65`, `127/128/129`, `tile±1`, multiple tiles, and the maximum length; cover `rows` below, equal to, and above the vector-core count; and cover positive-negative cancellation, mixed magnitudes, NaN/Inf, and the overflow contract. Compare against `torch.cumsum(x.float(), dim=-1)` and verify every cross-tile carry.

## Performance Gate

The dependency chain limits compute overlap, so primarily measure DMA overlap and per-tile overhead.

First pass PTO lowering and targeted correctness tests. Compare the complete end-to-end latency of the serial correctness implementation, the SIMD lane scan, one-stage and two-stage DMA, and the three-kernel split-row implementation. Report effective bandwidth, Vector/MTE time, core utilization, workspace bytes, and every launch.
