# SIMD Gather Transpose

## Applicability

Use when the dtype is 16/32 bit and source indexes can be represented within the range of a PTO SIMD index vector.

## PTO Kernel Structure

Use `T.simd.vci` to construct lane offsets, multiply them by the padded stride, add the column offset, reinterpret the result as an unsigned index vector, then use `T.simd.vgather2` and `T.simd.vsts`. Generate each `vector_id` within a static `T.Unroll`.

Use `testing/ascend/layout/test_ascend_l0_transpose.py` in the current repository to verify L0 transpose semantics, and use `tilelang/ascend/language/copy_op.py` and `src/ascend/op/copy.cc` to verify movement and layout constraints. These are not equivalent to a generic batched-transpose kernel. When reusing a current operator or creating a new implementation, specify task assignment, stride, dtype, tails, and UB footprint. Transpose adapters in the Skill must receive a kernel factory explicitly validated in the current repository.

## Accuracy Requirements

The index dtype must cover the maximum UB offset. The mask must match the lane count for the dtype. Unused tail lanes must not be written back to GM.

According to the current public interface, cover dtype, batch, stride, permitted alignment and remainder classes, and noncontiguous inputs. When the interface promises generic tail support, cover tile-1/tile/tile+1. `T.assume` may express only guarantees made by the caller. Compare output against a `torch.permute`/`transpose` reference, and use canaries to check for out-of-bounds access.

## Performance Requirements

Inspect the gather count, index recomputation, and unrolled code size in generated code. Reuse index preparation outside the column loop.

First pass targeted PTO accuracy tests, then compare DMA, SIMD gather, and SIMT fallback under identical conditions. Record latency, effective GM bandwidth, scalar-indexing overhead, UB occupancy, and generated-code length. Do not claim that one path is faster without measurement.
