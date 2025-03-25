# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import itertools
import tilelang
import tilelang.language as T
import torch
from tilelang.autotuner import autotune, jit


def get_configs(M, N, K):
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]

    _configs = list(
        itertools.product(block_M, block_N, block_K, num_stages, thread_num, enable_rasterization))

    return [{
        "block_M": c[0],
        "block_N": c[1],
        "block_K": c[2],
        "num_stages": c[3],
        "thread_num": c[4],
        "enable_rasteration": c[5],
    } for c in _configs]


def ref_program(A, B, BlockMask, C):
    batch_M = A.shape[0] // block_M
    batch_N = B.shape[1] // block_N
    batch_K = A.shape[1] // block_K

    for i in range(batch_M):
        for j in range(batch_N):
            accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=A.device)
            for k in range(batch_K):
                if BlockMask[i, j, k]:
                    accu += A[i*block_M:(i+1)*block_M, k*block_K:(k+1)*block_K].to(torch.float32) @ \
                           B[k*block_K:(k+1)*block_K, j*block_N:(j+1)*block_N].to(torch.float32)
            C[i * block_M:(i + 1) * block_M, j * block_N:(j + 1) * block_N] = accu.to(torch.float16)


def get_best_config(M, N, K):

    @autotune(
        configs=get_configs(M, N, K),
        keys=["block_M", "block_N", "block_K", "num_stages", "thread_num", "enable_rasteration"],
        warmup=3,
        rep=20,
    )
    @jit(out_idx=[-1], ref_prog=ref_program)
    def kernel(block_M=None,
               block_N=None,
               block_K=None,
               num_stages=None,
               thread_num=None,
               enable_rasteration=None):
        return blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages, thread_num,
                                  enable_rasteration)

    return kernel()


def blocksparse_matmul(M,
                       N,
                       K,
                       block_M,
                       block_N,
                       block_K,
                       num_stages,
                       thread_num,
                       enable_rasteration,
                       dtype="float16",
                       accum_dtype="float"):

    block_mask_shape = (M // block_M, N // block_N, K // block_K)

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            BlockMask: T.Tensor(block_mask_shape, "bool"),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if BlockMask[by, bx, k]:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned BlockSparse MatMul Benchmark")
    parser.add_argument("--m", type=int, default=1024, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity ratio (0-1)")
    parser.add_argument(
        "--use_autotune", action="store_true", default=False, help="Whether to use autotune")

    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k

    # Initialize input matrices
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    if args.use_autotune:
        best_latency, best_config, ref_latency = get_best_config(M, N, K)
        func = blocksparse_matmul(M, N, K, *best_config)
    else:
        func = blocksparse_matmul(M, N, K, 128, 128, 32, 2, 128, True)

    # Create block mask with desired sparsity
    block_M, block_N, block_K = 128, 128, 32  # default values if not using autotune
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > args.sparsity

    kernel = tilelang.compile(func, out_idx=-1)
    c = kernel(a, b, block_mask)

    # Verify result
    ref_c = torch.zeros_like(c)
    for i in range(M // block_M):
        for j in range(N // block_N):
            accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=a.device)
            for k in range(K // block_K):
                if block_mask[i, j, k]:
                    accu += (
                        a[i * block_M:(i + 1) * block_M, k * block_K:(k + 1) * block_K].to(
                            torch.float32) @ b[k * block_K:(k + 1) * block_K,
                                               j * block_N:(j + 1) * block_N].to(torch.float32))
            ref_c[i * block_M:(i + 1) * block_M,
                  j * block_N:(j + 1) * block_N] = accu.to(torch.float16)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
