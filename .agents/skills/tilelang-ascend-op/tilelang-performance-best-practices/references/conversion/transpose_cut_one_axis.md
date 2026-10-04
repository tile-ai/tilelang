# Single-Axis-Tiled Transpose

## Applicability

Use when one axis can reside fully in UB while the other must be tiled.

## PTO Kernel Structure

Treat the complete axis as a static inner footprint and index the tiled axis with Persistent tasks. When the complete axis is accessed repeatedly, move it only once within the current task. Write back only the valid output slice.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify movement and layout constraints. These are not equivalent to a generic batched-transpose kernel. When reusing a current operator or creating a new implementation, specify task assignment, stride, dtype, tails, and UB footprint. Transpose adapters in the Skill must receive a kernel factory explicitly validated in the current repository.

## Accuracy Requirements

Cover combinations of tails on both the complete and tiled axes.

According to the current public interface, cover dtype, batch, stride, permitted alignment and remainder classes, and noncontiguous inputs. When the interface promises generic tail support, cover tile-1/tile/tile+1. `T.assume` may express only guarantees made by the caller. Compare output against a `torch.permute`/`transpose` reference, and use canaries to check for out-of-bounds access.

## Performance Requirements

Compare copy count and wasted UB capacity against two-axis tiling. Automatically fall back to two-axis tiling when the complete axis is too large.

First pass targeted PTO accuracy tests, then compare DMA, SIMD gather, and SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB occupancy, and generated-code length. Do not claim that one path is faster without measurement.
