# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
import numpy as np

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
def bitnet_158_int8xint2_decode(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    fast_decoding=True,
    n_partition=4,
    reduce_thread=32,
):
    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"
    storage_nbit = 8
    num_bits = 2
    A_shape = (M, K)
    B_shape = (N, K // storage_nbit * num_bits)
    C_shape = (M, N)

    num_elems_per_byte = 4
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    micro_size_k_compressed = micro_size_k // num_elems_per_byte
    storage_dtype = T.int8
    block_K = reduce_thread * micro_size_k

    use_dp4a = True
    dp4a_size = 4

    @T.prim_func
    def kernel(
        A: T.Buffer(A_shape, in_dtype),
        B: T.Buffer(B_shape, storage_dtype),
        C: T.Buffer(C_shape, out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, n_partition),
            M,
            threads=(reduce_thread, n_partition),
        ) as (
            bx,
            by,
        ):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.import_source(decode_i2s_to_i8s)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k_compressed):
                    B_quant_local[v] = B[
                        bx * n_partition + ni,
                        ko * (reduce_thread * micro_size_k_compressed) + kr * micro_size_k_compressed + v,
                    ]

                T.call_extern(
                    "handle",
                    "decode_i2u_to_i8s",
                    T.address_of(B_quant_local[0]),
                    T.address_of(B_dequantize_local[0]),
                )

                if use_dp4a:
                    for ki in T.serial(micro_size_k // dp4a_size):
                        T.dp4a(
                            A_local[ki * dp4a_size],
                            B_dequantize_local[ki * dp4a_size],
                            accum_res[0],
                        )
                else:
                    for ki in T.serial(micro_size_k):
                        accum_res[0] += A_local[ki] * B_dequantize_local[ki]

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                C[by, bx * n_partition + ni] = reduced_accum_res[0]

    return kernel


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


def assert_bitnet_158_int8xint2_decode_correctness(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding=True):
    program = bitnet_158_int8xint2_decode(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding)
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
    assert_bitnet_158_int8xint2_decode_correctness(1, 256, 256, T.int8, T.int32, T.int32)
