import argparse

import tilelang
import tilelang.language as T

from tilelang.layout import make_cutlass_metadata_layout
from tilelang.utils.sparse import randn_semi_sparse
from tilelang.utils.tensor import torch_assert_close

from triton.testing import do_bench

import torch

torch.manual_seed(42)

DEFAULT_CONFIG = {  # take best config from autotune script
    "4090": {
        T.float: {
            "block_M": 128,
            "block_N": 64,
            "block_K": 64,
            "num_stages": 1,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
        T.float16: {
            "block_M": 256,
            "block_N": 128,
            "block_K": 64,
            "num_stages": 2,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
    },
    "h20": {
        T.float: {
            "block_M": 128,
            "block_N": 64,
            "block_K": 128,
            "num_stages": 3,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
        T.float16: {
            "block_M": 128,
            "block_N": 64,
            "block_K": 128,
            "num_stages": 3,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
    },
}

ARCH_INFO = {"8.0": (16, "int16"), "8.9": (16, "int16"), "9.0": (8, "uint8")}


@tilelang.jit(out_idx=[-1])
def matmul_sp_fp16_custom_compress(
    M, N, K, accum_dtype, block_M, block_N, block_K, num_stages, thread_num, policy, enable_rasterization, use_cutlass_layout
):
    e_factor, e_dtype = (16, T.int16)

    @T.prim_func
    def gemm_sp_fp16_custom_compress(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), e_dtype),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            if use_cutlass_layout:
                T.annotate_layout(
                    {
                        E: make_cutlass_metadata_layout(E, mma_dtype=T.float16, arch="8.0", block_k=block_K),
                        E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=T.float16, arch="8.0", block_k=block_K),
                    }
                )
            T.clear(C_local)
            T.disable_warp_group_reg_alloc()
            T.use_swizzle(panel_size=10, enable=enable_rasterization)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp_v2(A_shared, E_shared, B_shared, C_local, False, False, policy=policy)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_sp_fp16_custom_compress


def torch_compress(dense):
    """
    A naive compression function, where each 4-bit meta matches 4 elements in original matrix in row major layout.
    """
    if dense.dim() != 2:
        raise RuntimeError(f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor")

    m, k = dense.shape

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(f"Number of rows of dense matrix {m} must be divisible by 16")
    else:
        if m % 32 != 0:
            raise RuntimeError(f"Number of rows of dense matrix {m} must be divisible by 32")
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}")

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, _m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, _m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = meta_n[:, :, 0] | (meta_n[:, :, 1] << 4) | (meta_n[:, :, 2] << 8) | (meta_n[:, :, 3] << 12)
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    return (sparse, meta)


def decode_metadata(meta: torch.Tensor) -> torch.Tensor:
    assert meta.dtype is torch.int16
    groups_per_meta = 16 // 4  # 4 groups per uint16
    out = []
    for g in range(groups_per_meta):
        group_bits = (meta >> (g * 4)) & 0xF
        idx0 = group_bits & 0x3
        idx1 = (group_bits >> 2) & 0x3
        out.append(torch.stack([idx0, idx1], dim=-1))
    return torch.concat(out, dim=-1).view(meta.shape[0], -1)


@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    },
)
def compress_kernel(M, K, block_M, block_K, dtype, use_cutlass_layout):
    e_factor, e_dtype = ARCH_INFO["8.0"]
    e_K = K // e_factor
    elem, group = 2, 4

    assert M % block_M == 0, "M must be divisible by block_M"
    assert K % block_K == 0, "K must be divisible by block_K"
    assert K % e_factor == 0, "K must be divisible by e_factor"
    assert block_K % e_factor == 0, "block_K must be divisible by e_factor"

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        A_sp: T.Tensor((M, K // 2), dtype),
        E: T.Tensor((M, e_K), e_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(K, block_K), threads=block_M) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_sp_shared = T.alloc_shared((block_M, block_K // 2), dtype)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            if use_cutlass_layout:
                T.annotate_layout(
                    {
                        E: make_cutlass_metadata_layout(E, mma_dtype=T.float16, arch="8.0", block_k=block_K),
                        E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=T.float16, arch="8.0", block_k=block_K),
                    }
                )
            T.clear(A_sp_shared)
            T.clear(E_shared)
            # TODO: alloc_var seems buggy here
            non_zero_cnt = T.alloc_var(dtype=T.uint8)
            non_zero_elt_log_idx = T.alloc_shared((elem,), dtype=T.uint8)
            T.copy(A[bx * block_M, by * block_K], A_shared)
            for tm in T.Parallel(block_M):
                for g_i in range(0, block_K // group):
                    a_k = g_i * group
                    non_zero_cnt = 0
                    for i in range(elem):
                        non_zero_elt_log_idx[i] = 0
                    for i in range(group):
                        val = A_shared[tm, a_k + i]
                        if val != 0.0:
                            non_zero_elt_log_idx[non_zero_cnt] = i
                            A_sp_shared[tm, a_k // 2 + non_zero_cnt] = val
                            non_zero_cnt += 1
                    # TODO: use T.device_assert(non_zero_cnt <= 2) after rebasing main
                    if non_zero_cnt == 1 and non_zero_elt_log_idx[0] == 3:
                        non_zero_elt_log_idx[0] = 0
                        non_zero_elt_log_idx[1] = 3
                        A_sp_shared[tm, a_k // 2 + 1] = A_sp_shared[tm, a_k // 2]
                        A_sp_shared[tm, a_k // 2] = 0.0
                    elif non_zero_cnt == 1:
                        A_sp_shared[tm, a_k // 2 + 1] = 0
                        non_zero_elt_log_idx[1] = 3
                    for i in T.serial(elem):
                        val = non_zero_elt_log_idx[i]
                        E_shared[tm, a_k // e_factor] |= T.shift_left(val, 4 * (g_i % (e_factor // group)) + 2 * i)
            T.copy(A_sp_shared, A_sp[bx * block_M, by * block_K // 2])
            T.copy(E_shared, E[bx * block_M, by * block_K // e_factor])

    return kernel


def main():
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument("--use_cutlass_layout", action="store_true", help="Use cutlass layout for E tensor")
    parser.add_argument("--use_torch_compressor", action="store_true", help="Use torch sparse for reference")
    parser.add_argument("--accum_dtype", type=str, default=T.float, choices=[T.float, T.float16], help="Accumulation datatype")
    parser.add_argument("--cfg", type=str, choices=["4090"], default="4090")
    args = parser.parse_args()
    kernel = matmul_sp_fp16_custom_compress(
        args.m, args.n, args.k, args.accum_dtype, **DEFAULT_CONFIG[args.cfg][args.accum_dtype], use_cutlass_layout=args.use_cutlass_layout
    )

    a = randn_semi_sparse(args.m, args.k, device="cuda", dtype=torch.half)
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.half)

    if args.use_torch_compressor:
        assert not args.use_cutlass_layout, "torch sparse must be used with naive layout"
        a_sparse, e = torch_compress(a)
    else:
        a_sparse, e = compress_kernel(args.m, args.k, 32, 32, T.float16, use_cutlass_layout=args.use_cutlass_layout)(a)

    c = kernel(a_sparse, e, b)

    ref_c = a @ b

    assert not c.isnan().any(), "Reference result contains NaNs, please report an issue"
    torch_assert_close(c, ref_c.to(c.dtype), rtol=1e-3, atol=1e-3)
    print(f"Precision check passed. Max diff: {(c - ref_c).abs().max()}, Mean diff: {(c - ref_c).abs().mean()}")

    latency = do_bench(lambda: kernel(a_sparse, e, b))
    ref_latency = do_bench(lambda: a @ b)

    total_flops = 2 * args.m * args.n * args.k
    tflops = total_flops / latency / 1e9
    ref_tflops = total_flops / ref_latency / 1e9
    print(f"Sparse TFLOPS: {tflops:.2f}, Latency: {latency / 1e3} s")
    print(f"Reference TFLOPS: {ref_tflops:.2f}, Latency: {ref_latency / 1e3:} s")


if __name__ == "__main__":
    main()
