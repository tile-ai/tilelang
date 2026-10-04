# SIMD Lane-Parallel Scan

## Applicability

Use `log2(lanes)` shift-add stages within a vector to replace a serial lane loop.

## TileLang/PTO Implementation

Use PTO-validated SIMD gather/index primitives to construct statically shifted vectors with `offset=1,2,4...`; at each stage, update only lanes where `lane>=offset`. Continue to chain vectors through an fp32 carry. This path must first pass an independent lowering microtest.

The correctness baseline is `references/scan/templates/dav310/scan_base.py`: each row is owned by exactly one kernel task, the final contiguous axis is tiled, the fp32 carry remains resident, and GM/UB transfers use `T.copy`. The implementation does not call `T.cumsum` or depend on a grid barrier.

## Correctness Gate

The mask at every stage must prevent serial contamination across vectors. For the tail vector, extract only the last valid lane.

Cover `cols=1`, `63/64/65`, `127/128/129`, `tile±1`, multiple tiles, and the maximum length; cover `rows` below, equal to, and above the vector-core count; and cover positive-negative cancellation, mixed magnitudes, NaN/Inf, and the overflow contract. Compare against `torch.cumsum(x.float(), dim=-1)` and verify every cross-tile carry.

## Performance Gate

Measure the lane-scan instruction count, register pressure, and short-row overhead. Retain the streaming fallback if this path fails or provides no improvement.

First pass PTO lowering and targeted correctness tests. Compare the complete end-to-end latency of the serial correctness implementation, the SIMD lane scan, one-stage and two-stage DMA, and the three-kernel split-row implementation. Report effective bandwidth, Vector/MTE time, core utilization, workspace bytes, and every launch.
