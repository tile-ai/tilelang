# Regular Tensor Move

## Applicable Scenarios

Use this approach when a layout change can be implemented through contiguous rectangular reinterpretation and batched T.copy operations without element-wise gather.

## PTO Kernel Structure

After merging contiguous dimensions, call T.copy directly on rectangular slices. Use transpose=True only for memory levels and shapes validated by target examples. Fall back to a general transpose kernel in all other cases.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify transfer and layout restrictions. These files do not constitute a general batched-transpose kernel. When reusing the current operator or implementing a new one, define task assignment, strides, dtype, tail handling, and the UB footprint explicitly. The skill's transpose adapter requires an explicitly supplied kernel factory that has been validated in the current repository.

## Correctness Requirements

Validate source/destination strides, overlap, and in-place semantics. If in-place operation is unsupported, reject it explicitly at the interface layer.

According to the current public interface, cover dtype, batch, stride, supported alignment and remainder classes, and noncontiguous inputs. When the interface promises general tail handling, cover tile-1/tile/tile+1. T.assume may express only guarantees provided by the caller. Compare output against a torch.permute/transpose reference and use canaries to detect out-of-bounds accesses.

## Performance Requirements

Evaluate copy transactions, burst lengths, and effective bandwidth; do not judge performance solely by a reduction in DSL line count.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and the SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB usage, and generated-code length. Do not claim that a path is faster without measurements.
