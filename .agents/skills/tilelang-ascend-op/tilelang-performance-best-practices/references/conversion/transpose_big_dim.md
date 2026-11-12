# Large-Dimension Tiled Transpose

## Applicable Scenarios

Use two-dimensional tiling when either transposed dimension exceeds UB capacity.

## PTO Kernel Structure

Each task corresponds to one batch/x/y rectangle, and `T.Persistent` distributes tasks evenly across cores. Choose `block_x`/`block_y` so that inputs, outputs, padding, and any required multiversion buffers all fit within the UB budget.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` together with `src/ascend/op/copy.cc` to verify transfer and layout constraints. These are not equivalent to a general batched-transpose kernel. When reusing the current operator or implementing a new one, explicitly define task assignment, stride, dtype, tail blocks, and UB footprint. A transpose adapter bundled with the Skill requires an explicitly supplied kernel factory validated in the current repository.

## Correctness Requirements

Validate cross-tile boundaries, the final tile, and 64-bit total offsets. Dynamic strides must not be truncated to 32 bits.

According to the current public interface, cover dtype, batch, stride, permitted alignments and remainder classes, and noncontiguous inputs. When the interface promises general tail-block support, cover tile-1/tile/tile+1. `T.assume` may express only caller-guaranteed conditions. Compare outputs against a `torch.permute`/`transpose` reference and use canaries to detect out-of-bounds access.

## Performance Requirements

Search the aspect ratio of two-dimensional tiles and observe gather-instruction efficiency, copy bursts, and load balance.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB occupancy, and generated-code length. Do not claim that a path is faster without measurements.
