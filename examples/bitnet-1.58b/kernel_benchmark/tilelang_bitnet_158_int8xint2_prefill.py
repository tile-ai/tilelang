# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
from tilelang.intrinsics.mma_layout import (
    make_mma_swizzle_layout as make_swizzle_layout,
)
import numpy as np

from tilelang.intrinsics.mma_macro_generator import (
    INT4TensorCoreIntrinEmitter,
)
from tilelang.transform import simplify_prim_func

torch.manual_seed(42)

decode_i2s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2b = {e7,e6,e5,e4,e3,e2,e1,e0}
    // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
    uint const i2b = *reinterpret_cast<uint *>(_i2b);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;    // 1024
    static constexpr uint MEDIAN_NUM = 0x02020202;
#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsub4(i8s[i], MEDIAN_NUM);
    }
}
template <typename T1, typename T2>
__device__ void decode_i2u_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2b = {e7,e6,e5,e4,e3,e2,e1,e0}
    // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
    uint const i2b = *reinterpret_cast<uint *>(_i2b);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;    // 1024

#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
    }
}
"""


@simplify_prim_func
def bitnet_158_int8xint2_prefill(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    fast_decoding=True,
    block_row_warps=2,
    block_col_warps=2,
    warp_row_tiles=32,
    warp_col_tiles=32,
    chunk=64,
):
    """
    Create a TVM GPU prim_func implementing a block-tiled matrix multiply that multiplies dense A by compressed/interleaved low‑precision B (2-bit packed into int8 storage), decoding B to int8 on-chip and accumulating into C.

    The returned prim_func expects:
    - A: shape (M, K) with dtype `in_dtype` (T.float16 or T.int8).
    - B: compressed storage with shape (N, K/4) and int8 storage layout (packing 4 2-bit elements per byte).
    - C: output buffer shape (M, N) with dtype `out_dtype` (T.float16, T.float32, or T.int32).

    Details:
    - Builds a tiled, pipelined kernel using shared memory and warp-level MMA intrinsics (INT4TensorCoreIntrinEmitter). B is loaded from compressed storage, decoded to int8 in threads (via decode_i2u_to_i8s / decode_i2s_to_i8s), and dequantized into a shared buffer used by the MMA emitter.
    - Tiling parameters:
      - block_row_warps, block_col_warps: number of warps per block in row/col.
      - warp_row_tiles, warp_col_tiles: tiles per warp.
      - chunk: K-sized chunk per block (block_K).
      - micro sizes are fixed (16x16x16, except micro_k=32 when accum_dtype == T.int32).
    - Uses 2-stage pipelining by default to overlap loads and compute and applies a swizzle layout to improve L2 behavior.
    - Assertions: raises AssertionError if in_dtype or out_dtype are not among supported values.

    Parameters:
        M, N, K (int): Global matrix dimensions.
        in_dtype (str): Input and decoded B element dtype; T.float16 or T.int8.
        out_dtype (str): Output C dtype; one of T.float16, T.float32, T.int32.
        accum_dtype (str): Accumulator dtype used by MMA (e.g., T.int32).
        fast_decoding (bool): If True, enable the fast decoding path (affects which device decode is used).
        block_row_warps (int): Warps in block row dimension.
        block_col_warps (int): Warps in block column dimension.
        warp_row_tiles (int): Tiles per warp in row dimension.
        warp_col_tiles (int): Tiles per warp in column dimension.
        chunk (int): K-length per block (block_K).

    Returns:
        T.prim_func: A TVM prim_func implementing the described GPU kernel suitable for compilation and execution.
    """
    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if accum_dtype == T.int32:
        micro_size_k = 32

    num_elems_per_byte = 4
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    local_size_compressed = local_size // num_elems_per_byte

    shared_scope = "shared.dyn"
    storage_dtype = T.int8

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)  # int8 storage represents int4*2
    B_shape = (N, K // num_elems_per_byte)  # int8 storage represents int4*2
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    fragement_size_a = (micro_size_x * micro_size_k) // warp_size
    fragement_size_b = (micro_size_y * micro_size_k) // warp_size
    fragement_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = INT4TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, in_dtype),
        B: T.Buffer(B_shape, storage_dtype),
        C: T.Buffer((M, N), out_dtype),
    ):
        """
        GPU kernel entry that performs a blocked, pipelined matrix multiplication A @ B.T writing into C.

        This kernel:
        - Loads tiles of A and a compressed/interleaved representation of B from global memory into shared memory.
        - Decodes B's packed low-precision format (storage_dtype, e.g., 2-bit packed) into element values of `in_dtype` in shared memory via an external decode routine.
        - Uses Warp/MMA tiled fragments and an INT4/INT2-capable MMA emitter to compute accumulation across K in a pipelined fashion with configurable stages.
        - Writes accumulated tile results from shared memory back to global C with the expected block/micro-tile indexing.

        Parameters:
            A: Input matrix buffer of shape A_shape and element type `in_dtype`. Represents the MxK activations.
            B: Compressed/interleaved weight buffer of shape B_shape and storage type `storage_dtype`. Must contain B in the packed low-precision layout expected by the decode routine used by this kernel.
            C: Output buffer of shape (M, N) and type `out_dtype`; receives the resulting matrix (accumulated values are produced in `accum_dtype` and stored into C).

        Side effects:
            Writes results into C. Calls external device decode functions to expand B from its packed representation into shared memory before computation.
        """
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
            prelude=decode_i2s_to_i8s,
        ) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_frag = T.alloc_local((warp_rows * fragement_size_a), in_dtype)
            B_frag = T.alloc_local((warp_cols * fragement_size_b), in_dtype)
            C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size_c), accum_dtype)

            B_local = T.alloc_local([local_size_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([local_size], in_dtype)

            thread_bindings = T.get_thread_binding(0)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                    B_dequantize_shared: make_swizzle_layout(B_dequantize_shared),
                }
            )

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_frag)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K // num_elems_per_byte):
                    B_shared[j, k] = B[bx * block_N + j, ko * (block_K // num_elems_per_byte) + k]

                for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * local_size_compressed)):
                    for v in T.vectorized(0, local_size_compressed):
                        index = i * threads * local_size_compressed + thread_bindings * local_size_compressed + v
                        vi, vj = T.index_to_coordinates(index, B_shared_shape)
                        B_local[v] = B_shared[vi, vj]

                    T.call_extern(
                        "handle",
                        "decode_i2u_to_i8s",
                        T.address_of(B_local[0]),
                        T.address_of(B_dequantize_local[0]),
                    )

                    for v in T.vectorized(0, local_size):
                        index = i * threads * local_size + thread_bindings * local_size + v
                        vi, vj = T.index_to_coordinates(index, B_dequantize_shared_shape)
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_frag,
                        A_shared,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_frag,
                        B_dequantize_shared,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_frag, B_frag, C_frag)

            # Perform STMatrix
            mma_emitter.stmatrix(
                C_frag,
                C_shared,
            )

            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return main


def general_compress(lowprecision_weight, source_bits=4, storage_dtype=np.int8):
    elems_per_byte = 8 // source_bits
    if lowprecision_weight.dtype == np.float16:
        lowprecision_weight = lowprecision_weight.astype(dtype=np.int8)
    int8_weight = np.zeros(
        (
            *lowprecision_weight.shape[:-1],
            lowprecision_weight.shape[-1] // elems_per_byte,
        ),
        dtype=np.int8,
    )
    for j in range(lowprecision_weight.shape[-1] // elems_per_byte):
        for k in range(elems_per_byte):
            int8_weight[:, j] |= lowprecision_weight[:, j * elems_per_byte + k] << (source_bits * k)

    return int8_weight.view(storage_dtype)


# interleave weight numpy implementation
def interleave_weight(qweight, nbits=4, target_dtype=T.float16):
    assert target_dtype in [T.float16, T.int8]
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8 if target_dtype == T.int8 else 16
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    if nbits == 1 and target_dtype == T.int8:
        # special handling for 1b interleave
        n16_weight = new_qweight & np.int32(0xF0F00F0F)
        n16_weight |= ((new_qweight & np.int32(0x000000F0)) >> 4) << 16
        n16_weight |= ((new_qweight & np.int32(0x0000F000)) >> 12) << 24
        n16_weight |= ((new_qweight & np.int32(0x000F0000)) >> 16) << 4
        n16_weight |= ((new_qweight & np.int32(0x0F000000)) >> 24) << 12
        return n16_weight.view(np.int8)
    elif nbits == 2 and target_dtype == T.float16:
        n8_weight = new_qweight & np.int32(0xFF0000FF)
        n8_weight |= ((new_qweight & np.int32(0x0000FF00)) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(0x00FF0000)) >> 16) << 8
        return n8_weight.view(np.int8)
    elif nbits == 1 and target_dtype == T.float16:
        n8_weight = new_qweight & 0xF000000F
        n8_weight |= ((new_qweight & 0x000000F0) >> 4) << 8
        n8_weight |= ((new_qweight & 0x00000F00) >> 8) << 16
        n8_weight |= ((new_qweight & 0x0000F000) >> 12) << 24
        n8_weight |= ((new_qweight & 0x000F0000) >> 16) << 4
        n8_weight |= ((new_qweight & 0x00F00000) >> 20) << 12
        n8_weight |= ((new_qweight & 0x0F000000) >> 24) << 20

    return new_qweight.view(np.int8)


def assert_bitnet_158_int8xint2_prefill_correctness(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding=True):
    program = bitnet_158_int8xint2_prefill(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding)
    print(program)
    kernel = tilelang.compile(program)
    src_code = kernel.get_kernel_source()
    # src_code is the generated cuda source
    assert src_code is not None
    print(src_code)
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    qw = general_compress(B.cpu().numpy(), source_bits=2, storage_dtype=np.int8)
    qw = interleave_weight(qw, 2, target_dtype=in_dtype)
    qw = torch.from_numpy(qw).to(device="cuda")

    kernel(A, qw, C)
    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))

    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    assert_bitnet_158_int8xint2_prefill_correctness(256, 256, 256, T.int8, T.int32, T.int32)
