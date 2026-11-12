# Transpose and Layout Conversion

## Applicable Scenarios

Select DMA, SIMD gather, or SIMT according to contiguity, dtype byte width, tile footprint, and target layout.

## PTO Kernel Structure

First merge adjacent contiguous dimensions and select block_x/block_y. Use T.StridedTensor to express dynamic input strides and a contiguous T.Tensor for output. Read [batched_transpose_asc.py](code/batched_transpose_asc.py) for complete host tiling, multi-axis Persistent mapping, UB padding, and SIMD/SIMT kernel paths. Generate static indexing variants through a factory for complex layouts.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify transfer and layout restrictions. When reusing the reference or implementing a new operator, define task assignment, strides, dtype, tail handling, and the UB footprint explicitly. The skill's transpose adapter requires an explicitly supplied kernel factory that has been validated in the current repository.

## Correctness Requirements

The conversion must not alter element values or duplicate/omit elements. Use exact element-wise comparison for low-precision and integer data. Use the project's existing floating-point tolerance only when a numerical conversion occurs at the same time.

According to the current public interface, cover dtype, batch, stride, supported alignment and remainder classes, and noncontiguous inputs. When the interface promises general tail handling, cover tile-1/tile/tile+1. T.assume may express only guarantees provided by the caller. Compare output against a torch.permute/transpose reference and use canaries to detect out-of-bounds accesses.

## Performance Requirements

Prioritize reducing GM transactions and address calculations. Control launch/initialization overhead for small shapes, and optimize effective bandwidth for large shapes.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and the SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB usage, and generated-code length. Do not claim that a path is faster without measurements.
