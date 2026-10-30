# Transpose Fusion

## Applicable Scenarios

When elementwise, cast, scale, or quantization operations occur before or after a transpose, fusion can eliminate an intermediate GM round trip.

## PTO Kernel Structure

Execute the fused expression before writing back `out_ub`; place regular contiguous expressions in `T.SimdVF`. Keep transpose indexing layered separately from numerical operations, and let the factory decide whether to generate a fused variant.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` together with `src/ascend/op/copy.cc` to verify transfer and layout constraints. These are not equivalent to a general batched-transpose kernel. When reusing the current operator or implementing a new one, explicitly define task assignment, stride, dtype, tail blocks, and UB footprint. A transpose adapter bundled with the Skill requires an explicitly supplied kernel factory validated in the current repository.

## Correctness Requirements

Strictly preserve the reference's operation and cast order. Separately validate transpose-only, elementwise-only, and fused results.

According to the current public interface, cover dtype, batch, stride, permitted alignments and remainder classes, and noncontiguous inputs. When the interface promises general tail-block support, cover tile-1/tile/tile+1. `T.assume` may express only caller-guaranteed conditions. Compare outputs against a `torch.permute`/`transpose` reference and use canaries to detect out-of-bounds access.

## Performance Requirements

Keep the fusion only when it reduces total GM bytes without introducing a fallback due to register or UB pressure.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB occupancy, and generated-code length. Do not claim that a path is faster without measurements.
