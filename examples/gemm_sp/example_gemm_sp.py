import argparse

import tilelang
import tilelang.language as T

from tilelang.utils.sparse import compress, randn_semi_sparse, get_e_factor
from tilelang.profiler import do_bench

import torch


@tilelang.jit(out_idx=[-1])
def matmul_sp_fp16(M, N, K, accum_dtype, e_dtype, block_M, block_N, block_K, num_stages, thread_num, policy, enable_rasterization):
    e_factor = get_e_factor(T.float16, e_dtype)

    @T.prim_func
    def gemm_sp_fp16(
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

            T.clear(C_local)
            T.disable_warp_group_reg_alloc()
            T.use_swizzle(panel_size=10, enable=enable_rasterization)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, transpose_A=False, transpose_E=False, transpose_B=False, policy=policy)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_sp_fp16


def main(
    M=1024,
    N=1024,
    K=1024,
    accum_dtype=T.float,
    e_dtype=T.int16,
    block_M=128,
    block_N=128,
    block_K=64,
    num_stages=2,
    thread_num=128,
    policy=T.GemmWarpPolicy.Square,
    enable_rasterization=True,
):
    kernel = matmul_sp_fp16(M, N, K, accum_dtype, e_dtype, block_M, block_N, block_K, num_stages, thread_num, policy, enable_rasterization)

    a = randn_semi_sparse(M, K, device="cuda", dtype=torch.half)
    b = torch.randn(K, N, device="cuda", dtype=torch.half)

    a_sparse, e = compress(a, meta_dtype=e_dtype.as_torch())
    c = kernel(a_sparse, e, b)

    ref_c = a @ b

    assert not c.isnan().any(), "Reference result contains NaNs, please report an issue"
    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-2, atol=1e-2)
    print(f"Precision check passed. diff: {(c - ref_c).abs().mean()}")

    latency = do_bench(lambda: kernel(a_sparse, e, b))
    ref_latency = do_bench(lambda: a @ b)

    total_flops = 2 * M * N * K
    tflops = total_flops / latency / 1e9
    ref_tflops = total_flops / ref_latency / 1e9
    print(f"Sparse TFLOPS: {tflops:.2f}, Latency: {latency / 1e3} s")
    print(f"Reference TFLOPS: {ref_tflops:.2f}, Latency: {ref_latency / 1e3:} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse FP16 MatMul Example")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument(
        "--e_dtype",
        default=T.int16,
        choices=[T.int8, T.int16, T.int32],
        help="Data type for metadata E, which controls the sparsity pattern. Note that int8 and int32 are only supported on sm90+",
    )
    parser.add_argument("--accum_dtype", default=T.float, choices=[T.float, T.float16], help="Accumulation datatype")
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=256)
    parser.add_argument("--block_K", type=int, default=128)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--thread_num", type=int, default=256)
    args = parser.parse_args()
    main(
        M=args.m,
        N=args.n,
        K=args.k,
        accum_dtype=args.accum_dtype,
        e_dtype=args.e_dtype,
        block_M=args.block_M,
        block_N=args.block_N,
        block_K=args.block_K,
        num_stages=args.num_stages,
        thread_num=args.thread_num,
    )
