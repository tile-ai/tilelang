import tilelang
import tilelang.language as T

import torch


@tilelang.jit(execution_backend="nvrtc")
def matmul(M, N, K, block_M, block_N, block_K, dtype="float32", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope='shared')
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope='shared')
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j, k in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return gemm


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    BLOCK_SIZE = 32
    jit_kernel = matmul(M, N, K, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

    jit_kernel(a, b, c)


import torch

tile_add = solve

sizes = [64 * 2**p for p in range(0, 5)]
y1 = []
y2 = []
y3 = []

device = 'mps'

for size in sizes:
    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)
    c = torch.zeros(size, size, dtype=torch.float32, device=device)

    tile_add(a, b, c, size, size, size)

    # currently breaks when size > 1024
    assert torch.allclose(c, a @ b), f'size={size}, a={a}, b={b}, c={c}'

print('pass')
