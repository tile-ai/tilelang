# UB Resident Scan

## Applicability

Use this design when the entire row and its fp32 workspace fit in UB.

## TileLang/PTO Implementation

Transfer the entire row at once, complete the prefix operation in UB/a fragment, and then write back once. Allocate padding according to dtype lane alignment, but process only valid elements.

See `references/scan/templates/dav310/scan_base.py` for the correctness baseline: each row is owned by exactly one kernel task, tiled along the final contiguous axis, with an fp32 carry kept resident and T.copy used for GM/UB transfers. The implementation does not call T.cumsum or depend on a grid barrier.

## Accuracy Gates

Verify that the final valid lane, rather than padding, is used as the output.

Cover cols=1, 63/64/65, 127/128/129, tile±1, multiple tiles, and the maximum length; rows below, equal to, and above the number of vector cores; and positive-negative cancellation, mixtures of large and small magnitudes, NaN/Inf, and the overflow contract. Compare against torch.cumsum(x.float(), dim=-1), and check every carry across tile boundaries.

## Performance Gates

This design is suitable for short and medium rows. Compare launch overhead, copies, and UB usage between full-load and streaming implementations.

First pass PTO lowering and targeted accuracy tests. Compare the complete end-to-end latency of serial correctness, SIMD lane scan, 1/2-stage DMA, and the three-kernel split-row design. Report effective bandwidth, Vector/MTE time, core utilization, workspace bytes, and every launch.
