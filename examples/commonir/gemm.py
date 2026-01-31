# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

import torch
import torch_npu
device = torch.npu.current_device()
dtype = torch.float16

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
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

    return gemm


def main():
    func = matmul(1024, 1024, 1024, 128, 128, 32)
    kernel = tilelang.compile(func, target='commonir')
    SIZEALL = 1024

    torch.manual_seed(0)
    a = torch.rand((SIZEALL, SIZEALL), dtype=dtype, device=device) - 0.5
    b = torch.rand((SIZEALL, SIZEALL), dtype=dtype, device=device) - 0.5
    result = torch.zeros((SIZEALL, SIZEALL), dtype=dtype, device=device)

    kernel(a, b, result)
    golden = a @ b
    # print(f"result is {result}, golden is {golden}")
    torch.testing.assert_close(result, golden, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    main()