# Small-Shape Transpose

## Applicable Scenarios

Use this pattern when there are few independent tiles, the total byte count is small, and launch and indexing overhead dominate.

## PTO Kernel Structure

Use `min(vector core count, tile count)`. When the data fits in one UB tile, use one `T.copy` to transfer it in, one vector/SIMT transpose, and one writeback; do not enable multiple stages. Generate a small number of static variants in the factory according to dtype and remainder class.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` together with `src/ascend/op/copy.cc` to verify transfer and layout constraints. These are not equivalent to a general batched-transpose kernel. When reusing the current operator or implementing a new one, explicitly define task assignment, stride, dtype, tail blocks, and UB footprint. A transpose adapter bundled with the Skill requires an explicitly supplied kernel factory validated in the current repository.

## Correctness Requirements

Cover one element, one row, one column, the 64/128 boundaries, and `batch=1`.

According to the current public interface, cover dtype, batch, stride, permitted alignments and remainder classes, and noncontiguous inputs. When the interface promises general tail-block support, cover tile-1/tile/tile+1. `T.assume` may express only caller-guaranteed conditions. Compare outputs against a `torch.permute`/`transpose` reference and use canaries to detect out-of-bounds access.

## Performance Requirements

Compare one core against a small number of cores. Avoid launching many idle cores or constructing complex index tables for small shapes.

First pass targeted PTO correctness tests, then compare DMA, SIMD gather, and SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB occupancy, and generated-code length. Do not claim that a path is faster without measurements.
