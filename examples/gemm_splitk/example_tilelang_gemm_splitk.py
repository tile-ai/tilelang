# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
from tvm import DataType


def matmul(M, N, K, block_M, block_N, block_K, split_k, dtype="float16", accum_dtype="float"):

    splitK = K // split_k

    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if bz == 0:
                # fuse the zero initialization kernel
                for i, j in T.Parallel(block_M, block_N):
                    m, n = by * block_M + i, bx * block_N + j
                    C[m, n] = T.cast(0, dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
                T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            if DataType(dtype).bits == 16:
                for i, j in T.Parallel(block_M, block_N // 2):
                    m, n = by * block_M + i, bx * block_N + j * 2
                    # vectorized atomic
                    T.atomic_addx2(C[m, n], C_shared[i, j * 2])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

    return main


program = matmul(1024, 1024, 1024, 128, 128, 32, 4)

kernel = tilelang.compile(program)

print(kernel.get_kernel_source())

import torch

a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()
c = torch.zeros(1024, 1024).cuda().half()
kernel(a, b, c)

ref_c = a @ b

print(c)
print(ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
