# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.testing
import tilelang.language as T
import torch


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
                    bx,
                    by,
                ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            # changing num_stages to 0 gives correct results
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, ko * block_K], A_shared)

                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_pipeline_test(N, block_M=128, block_N=128, block_K=32):
    func = matmul(N, N, N, block_M, block_N, block_K)
    jit_kernel = tilelang.JITKernel(func, out_idx=[2], target="cuda")

    torch.manual_seed(0)
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)

    ref_c = a @ b.T
    c = jit_kernel(a, b)

    tilelang.testing.torch_assert_close(c, ref_c, rtol=1e-2, atol=0.2)


def test_pipeline_large_matrix():
    """Test pipeline stages with large matrix multiplication (8192x8192)"""
    run_gemm_pipeline_test(8192)


def test_pipeline_small_matrix():
    """Test pipeline stages with smaller matrix multiplication"""
    run_gemm_pipeline_test(1024)


if __name__ == "__main__":
    tilelang.testing.main()
