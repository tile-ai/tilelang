# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang.testing
import tilelang.language as T


def atomic_add_program(K, M, N, block_M, block_N, dtype="float"):

    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                   A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_add


def run_atomic_add(K, M, N, block_M, block_N, dtype="float32"):
    program = atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    kernel = tilelang.compile(program)
    # print(kernel.get_kernel_source())
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] += A[k, i, j]

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B)


def test_atomic_add():
    run_atomic_add(8, 128, 128, 32, 32)


if __name__ == "__main__":
    tilelang.testing.main()
