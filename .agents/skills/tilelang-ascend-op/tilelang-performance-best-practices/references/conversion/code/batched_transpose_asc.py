import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores


@tilelang.jit()
def get_batched_transpose_kernel_asc(
    shape_x_mod_128: int,
    shape_y_mod_128: int,
    dtype: T.dtype,
    block_x: int,
    block_y: int,
):
    assert shape_x_mod_128 in (0, 64) and shape_y_mod_128 in (0, 64)
    assert block_x in (64, 128) and block_y in (32, 64, 128)

    num_batches = T.dynamic("num_batches")
    shape_x = T.dynamic("shape_x")
    shape_y = T.dynamic("shape_y")
    stride_x = T.dynamic("stride_x")

    n_cores = get_num_vec_cores()
    num_threads = 256
    vector_elems = 256 // dtype.bytes
    num_x_vectors = (block_x + vector_elems - 1) // vector_elems
    padded_block_x = block_x if dtype.bytes == 1 else num_x_vectors * vector_elems
    # Break the power-of-two row stride used by indexed UB loads. Keep the
    # padding at one 32-byte data block so the 2-D GM->UB copy can encode the
    # destination gap without falling back to scalar row copies.
    ub_row_pad = 0 if dtype.bytes == 1 else 32 // dtype.bytes
    padded_block_y = block_y if dtype.bytes == 1 else ((block_y + vector_elems - 1) // vector_elems) * vector_elems + ub_row_pad
    index_dtype = "int32" if dtype.bytes == 4 else "int16"
    unsigned_index_vector_dtype = "uint32x64" if dtype.bytes == 4 else "uint16x128"
    mask_pattern = "PAT_ALL" if block_x % vector_elems == 0 else f"PAT_VL{block_x % vector_elems}"
    num_x_tiles = shape_x // block_x
    num_y_tiles = shape_y // block_y

    @T.prim_func
    def batched_transpose_kernel_asc(
        x: T.StridedTensor[(num_batches, shape_x, shape_y), (shape_x * stride_x, stride_x, 1), dtype],
        out: T.Tensor[(num_batches, shape_y, shape_x), dtype],
    ):
        with T.Kernel(n_cores) as core_id:
            # Pad UB rows to complete SIMD registers. A masked vector access
            # may still touch a register-width footprint in UB.
            x_ub = T.alloc_shared((block_x, padded_block_y), dtype)
            out_ub = T.alloc_shared((block_y, padded_block_x), dtype)

            T.assume(shape_x % block_x == 0)
            T.assume(shape_y % block_y == 0)
            T.assume(stride_x % 4 == 0)

            for pid_batch, pid_x, pid_y in T.Persistent(
                [num_batches, num_x_tiles, num_y_tiles],
                n_cores,
                core_id,
                group_size=1,
                num_stages=1,
            ):
                T.copy(
                    x[pid_batch, pid_x * block_x, pid_y * block_y],
                    x_ub[:block_x, :block_y],
                )
                if dtype.bytes == 1:
                    # CANN indexed gather/scatter pairs a 256-lane FP8 vector
                    # with a 128-lane index vector, which cannot express this
                    # byte-wise transpose. Keep the verified FP8 path.
                    with T.SimtVF(threads=num_threads):
                        for i, j in T.Parallel(block_x, block_y):
                            out_ub[j, i] = x_ub[i, j]
                else:
                    with T.SimdVF():
                        mask = T.simd.pset(dtype.bits, mask_pattern)
                        base_indices = T.simd.alloc_local((num_x_vectors,), index_dtype)

                        for vector_id in T.Unroll(num_x_vectors, explicit=True):
                            if dtype.bytes == 4:
                                offsets = T.simd.vci(T.int32(vector_id * vector_elems), index_dtype)
                                base_indices[vector_id] = T.simd.vmuls(offsets, T.int32(padded_block_y), mask)
                            else:
                                offsets = T.simd.vci(T.int16(vector_id * vector_elems), index_dtype)
                                base_indices[vector_id] = T.simd.vmuls(offsets, T.int16(padded_block_y), mask)

                        for j in T.serial(block_y):
                            for vector_id in T.Unroll(num_x_vectors, explicit=True):
                                if dtype.bytes == 4:
                                    signed_indices = T.simd.vadds(base_indices[vector_id], T.int32(j), mask)
                                else:
                                    signed_indices = T.simd.vadds(base_indices[vector_id], T.int16(j), mask)
                                indices = T.reinterpret(signed_indices, unsigned_index_vector_dtype)
                                values = T.simd.vgather2(x_ub[0, 0], indices, mask)
                                T.simd.vsts(out_ub[j, vector_id * vector_elems], values, mask)
                T.copy(out_ub[:block_y, :block_x], out[pid_batch, pid_y * block_y, pid_x * block_x])

    return batched_transpose_kernel_asc
