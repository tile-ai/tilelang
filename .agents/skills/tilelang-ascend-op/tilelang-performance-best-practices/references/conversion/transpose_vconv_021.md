# 0-2-1 Three-Dimensional Transformation

## Applicable Scenarios

Use this pattern to swap the last two axes of a three-dimensional layout, commonly for batch/sequence/channel reordering.

## PTO Kernel Structure

Treat dimension 0 as the batch task and reuse a two-dimensional transpose for the last two dimensions. Generate `block_x`/`block_y` for fixed remainder classes. If dtype conversion occurs at the same time, perform it in UB using fp32/the target dtype while preserving conversion order.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` together with `src/ascend/op/copy.cc` to verify transfer and layout constraints. These are not equivalent to a general batched-transpose kernel. When reusing the current operator or implementing a new one, explicitly define task assignment, stride, dtype, tail blocks, and UB footprint. A transpose adapter bundled with the Skill requires an explicitly supplied kernel factory validated in the current repository.

## Correctness Requirements

Cover `size=1` for each of the three dimensions, dynamic batch, and tails on both transposed axes.

According to the current public interface, cover dtype, batch, stride, permitted alignments and remainder classes, and noncontiguous inputs. When the interface promises general tail-block support, cover tile-1/tile/tile+1. `T.assume` may express only caller-guaranteed conditions. Compare outputs against a `torch.permute`/`transpose` reference and use canaries to detect out-of-bounds access.

## Performance Requirements

Compare the GM bytes and latency of fused conversion against a separate transpose + cast.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB occupancy, and generated-code length. Do not claim that a path is faster without measurements.
