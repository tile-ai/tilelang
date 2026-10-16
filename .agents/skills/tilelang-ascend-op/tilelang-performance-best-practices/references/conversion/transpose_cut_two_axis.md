# Two-Axis Tiled Transpose

## Applicable Scenarios

Use this approach when neither transposed axis can reside completely in UB.

## PTO Kernel Structure

Use a two-dimensional tile grid. The input x_ub has shape block_x×padded_y, and the output out_ub has shape block_y×padded_x. Generate index vectors from static block sizes outside the SIMD region and reuse them.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify transfer and layout restrictions. These files do not constitute a general batched-transpose kernel. When reusing the current operator or implementing a new one, define task assignment, strides, dtype, tail handling, and the UB footprint explicitly. The skill's transpose adapter requires an explicitly supplied kernel factory that has been validated in the current repository.

## Correctness Requirements

Validate tail tiles at all four corners and cases where both axes are non-divisible.

According to the current public interface, cover dtype, batch, stride, supported alignment and remainder classes, and noncontiguous inputs. When the interface promises general tail handling, cover tile-1/tile/tile+1. T.assume may express only guarantees provided by the caller. Compare output against a torch.permute/transpose reference and use canaries to detect out-of-bounds accesses.

## Performance Requirements

Search block_x/block_y within the UB capacity, avoiding extremely narrow tiles that reduce DMA and SIMD utilization.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and the SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB usage, and generated-code length. Do not claim that a path is faster without measurements.
