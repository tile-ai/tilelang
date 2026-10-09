# PTO RoPE Optimization Index

## Implementation

Use `references/rope/code/rope_vf_common.py` as the baseline. The Python factory specializes layout, dim, dtype, and position offset. A one-dimensional `T.Kernel` assigns token × head tasks with grid stride. `T.copy` moves the input, sin, and cos into UB. The PTO-compilable `T.SimtVF`/`T.Parallel` path performs pairwise rotation in fp32 and converts to the output dtype only at the end.

Half-split and interleaved layouts must generate different kernels; do not branch dynamically in the hot loop. An arbitrary position tensor first requires validation of PTO lowering for scalar/scattered indexing. The current executable baseline uses contiguous positions with a compile-time position offset. This SimtVF path is for correctness regression only. A production path must implement lowerable SIMD intrinsics, cross-head reuse, or operator fusion, with hardware, version, commands, and raw performance results attached.

## Correctness

Cover `position=0`/maximum position, offset, odd/even token and head counts, dim boundaries, both layouts, in-place/out-of-place operation, and bf16/fp16/fp32. The reference computes sin/cos, multiply-add, and concatenation in fp32; convert output only at the final boundary.

## Performance

Compare reloading sin/cos for each head against a token-grouped resident version. Report GM bytes, Vector/MTE time, UB occupancy, and latency. Run the relevant complete test suite after every targeted correctness test passes.
