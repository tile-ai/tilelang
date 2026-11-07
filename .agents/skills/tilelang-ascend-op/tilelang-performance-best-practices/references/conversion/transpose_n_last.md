# N-last Layout Conversion

## Applicable Scenarios

Use this approach to move a specified dimension to or from the last position, preferably after merging the other contiguous dimensions.

## PTO Kernel Structure

The factory normalizes the input to outer×move×inner. When inner is contiguous, use a rectangular T.copy; use SIMD gather to swap move and inner. Fall back to SIMT when byte indexing cannot be expressed safely.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify transfer and layout restrictions. These files do not constitute a general batched-transpose kernel. When reusing the current operator or implementing a new one, define task assignment, strides, dtype, tail handling, and the UB footprint explicitly. The skill's transpose adapter requires an explicitly supplied kernel factory that has been validated in the current repository.

## Correctness Requirements

Validate rank, negative axes, dimensions of size 1, and noncontiguous strides.

According to the current public interface, cover dtype, batch, stride, supported alignment and remainder classes, and noncontiguous inputs. When the interface promises general tail handling, cover tile-1/tile/tile+1. T.assume may express only guarantees provided by the caller. Compare output against a torch.permute/transpose reference and use canaries to detect out-of-bounds accesses.

## Performance Requirements

Select vector execution or SIMT according to the inner length, and compare address calculations and GM bursts before and after merging dimensions.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and the SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB usage, and generated-code length. Do not claim that a path is faster without measurements.
