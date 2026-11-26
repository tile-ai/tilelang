# PTO Prefix Scan

## Applicability

Applies to inclusive scan, cummin/cummax, and variants that return indices. First fix the axis, direction, dtype, accumulation semantics, and output semantics.

## TileLang/PTO Implementation

Use row-owned streaming when enough rows are available; short rows may use full-load. Use the three-kernel local-prefix, chunk-total-scan, and fixup design only for a very small number of extremely long rows. Enable lane-parallel execution only after the corresponding SIMD gather/shift microbenchmarks pass.

See `references/scan/templates/dav310/scan_base.py` for the correctness baseline: each row is owned by exactly one kernel task, tiled along the final contiguous axis, with an fp32 carry kept resident and `T.copy` used for GM/UB transfers. The implementation does not call `T.cumsum` or depend on a grid barrier. This serial inner scan is for accuracy regression only; performance numbers without accompanying hardware, version, commands, and raw results must not be used to justify production dispatch.

## Accuracy Gates

Use an independent reference for cummin/cummax tie-breaking, NaN behavior, and index propagation.

Cover cols=1, 63/64/65, 127/128/129, tile±1, multiple tiles, and the maximum length; rows below, equal to, and above the number of vector cores; and positive-negative cancellation, mixtures of large and small magnitudes, NaN/Inf, and the overflow contract. Compare against torch.cumsum(x.float(), dim=-1), and check every carry across tile boundaries.

## Performance Gates

Establish dispatch for short, medium, and long rows; do not decide from a single shape. The production path must replace the serial inner scan with either a SIMD lane scan that has passed lowering or a split-row implementation that is faster end to end.

First pass PTO lowering and targeted accuracy tests. Compare the complete end-to-end latency of serial correctness, SIMD lane scan, 1/2-stage DMA, and the three-kernel split-row design. Report effective bandwidth, Vector/MTE time, core utilization, workspace bytes, and every launch.
