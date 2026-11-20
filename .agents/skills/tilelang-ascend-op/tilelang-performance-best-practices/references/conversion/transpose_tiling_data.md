# Transpose Configuration Selection

## Applicable Scenarios

Use a Python kernel factory to generate cacheable static configurations without introducing an additional host-side configuration structure.

## PTO Kernel Structure

The inputs are the shape remainder class, dtype, stride category, and layout; the outputs are block_x, block_y, num_cores, execution domain, and stage. Retain dynamic dimensions only in T.dynamic, and express hard constraints through T.assume.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify data-movement and layout constraints; these are not equivalent to a general-purpose batched-transpose kernel. When reusing an existing operator or creating a new implementation, explicitly define task assignment, stride, dtype, tail blocks, and the UB footprint. The skill's transpose adapter must explicitly receive a kernel factory that has been verified in the current repository.

## Accuracy Requirements

Run accuracy tests on both sides of every dispatch boundary, and ensure that the fallback covers all valid inputs.

Cover dtype, batch, stride, permitted alignment and remainder classes, and non-contiguous inputs according to the current public interface. When the interface promises general tail-block support, cover tile-1/tile/tile+1. T.assume may express only guarantees made by the caller. Compare outputs against the torch.permute/transpose reference, and use canaries to detect out-of-bounds accesses.

## Performance Requirements

Enumerate candidates offline and benchmark them on representative shapes; control the number of variants and the compilation-cache footprint.

First pass targeted PTO accuracy tests, then compare DMA, SIMD gather, and the SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB usage, and generated-code length; do not claim that any path is faster without measurements.
