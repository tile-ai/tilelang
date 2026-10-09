# Three-Stage Split Scan for Extremely Long Rows

## Applicability

Use this approach when there are very few rows, cols is extremely large, and a row-owned kernel leaves many cores idle.

## TileLang/PTO Implementation

In Kernel A, each core scans one chunk and writes the local prefix and an fp32 chunk total. Kernel B scans the chunk totals for each row. Kernel C adds the previous chunk's carry to each chunk. The host wrapper executes the three kernels sequentially.

For the correctness baseline, see `references/scan/templates/dav310/scan_base.py`: each row is owned by exactly one kernel task, the last contiguous axis is partitioned into tiles, the fp32 carry remains resident, and GM/UB transfers use `T.copy`. The implementation does not call `T.cumsum` or rely on a grid barrier.

## Correctness Gate

Use fp32 for chunk totals and fixup. Cover `chunk=1`, a short final chunk, and different partition counts.

Cover `cols=1`, 63/64/65, 127/128/129, tile±1, multiple tiles, and the maximum length; test row counts below, equal to, and above the number of vector cores; and cover positive-negative cancellation, mixed magnitudes, NaN/Inf, and the overflow contract. Compare against `torch.cumsum(x.float(), dim=-1)` and check every carry across tile boundaries.

## Performance Gate

Account for all three launches, the workspace, and two additional GM passes. Enable this approach only when the complete path outperforms the single-owner implementation.

First pass PTO lowering and targeted correctness tests. Compare the complete end-to-end latency of the serial-correctness baseline, SIMD lane scan, one-stage/two-stage DMA, and the three-kernel split-row approach. Report effective bandwidth, Vector/MTE time, core utilization, workspace bytes, and all launches.
