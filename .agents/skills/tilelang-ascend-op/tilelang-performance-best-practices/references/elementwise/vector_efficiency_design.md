# Element-Wise SIMD and Fusion

## Goals and Applicability

Use this approach for contiguous element-wise operations, quantization conversions, activations, and simple compound expressions.

## TileLang/PTO Implementation

Use T.SimdVF with T.Parallel for contiguous regular paths. Convert bf16/fp16 inputs to fp32 before transcendental functions. Keep single-use intermediate results in registers or the same UB tile, and fuse adjacent expressions to eliminate GM round trips. Use T.SimtVF only for complex indexing.

For the affine chain `a*x+b`, treat explicit `S.vmadd` and `vmul+vadd` as distinct lowering candidates. Do not assume that fast-math automatically fuses them. Confirm the instruction in generated PTO, then validate correctness and same-methodology performance separately.

For short records, pack multiple tokens into one vector, use pregenerated indices with `vgather2`/`vscatter`, and hoist constants and index patterns out of the hot loop. For common full-tile shapes, the Python wrapper may specialize the shape and unroll count; retain dynamic tail handling for other shapes. See [Indexed Short-Record Transformation](indexed_short_record.md) for the complete structure.

The implementation must use a one-dimensional T.Kernel. For pure Vector tasks, the core count must not exceed `min(confirmed_available_aiv_core_count(), independent_task_count)`. Use T.copy between GM and UB/L1. Use T.SimdVF and T.Parallel for contiguous regular computation. Use T.Persistent or T.Pipelined for cross-tile tasks, following a validated multi-version template and verifying the lowering.

## Correctness Gate

Strictly preserve the reference operation order, saturation/rounding mode, and output-cast boundary. Quantization coverage must include zero scales, extreme values, and the interface contract for nonfinite inputs.

Coverage must include the minimum shape, common shapes, maximum shape, tile-1/tile/tile+1, tails not aligned to 32B, zero, positive and negative extremes, and the NaN/Inf semantics specified by the interface. Do not obtain a passing result by widening tolerances, reducing reference precision, or skipping cases.

## Performance Gate

Measure vector-lane utilization, instruction count, GM bytes, and spills. Split fused expressions when fusion causes register pressure or code-size growth.

First run targeted correctness tests with `TILELANG_DEFAULT_TARGET=pto`, then run the complete relevant test suite. After all tests pass, measure performance with identical inputs, dtype, warmup, repeat count, device, and concurrency. Report kernel latency, effective GM bandwidth, UB usage, stage count, and the difference from the baseline.

## Executable Code and Evidence

See examples/ascend/example_simdvf_vecadd.py and examples/ascend/example_simdvf_per_token_cast_to_fp8.py for executable implementations.
