# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


func = matmul(1024, 1024, 1024, 128, 128, 32)

print(func)

kernel = tilelang.compile(func, out_idx=-1)

import torch

a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()

c = kernel(a, b)

ref_c = a @ b

print(c)
print(ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# Get CUDA Source
print(kernel.get_kernel_source())
