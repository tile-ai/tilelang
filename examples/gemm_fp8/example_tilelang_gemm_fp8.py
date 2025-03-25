# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_gemm_fp8(M, N, K, dtype):
    torch_dtype = map_torch_type(dtype)

    func = matmul(M, N, K, 128, 128, 64, dtype)

    kernel = tilelang.compile(func, out_idx=-1)

    a = torch.randn(M, K, dtype=torch.float16, device='cuda').to(dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch.float16, device='cuda').to(dtype=torch_dtype)

    c = kernel(a, b)

    ref_c = (a.half() @ b.half().T).to(dtype=torch_dtype)

    print(c)
    print(ref_c)

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


if __name__ == "__main__":
    test_gemm_fp8(1024, 1024, 1024, 'e4m3_float8')
    test_gemm_fp8(1024, 1024, 1024, 'e5m2_float8')
