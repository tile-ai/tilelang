# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type


def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype="float"):
    # for fp8 gemm, do one promote after 4 wgmma inst, i.e. block_K = 128.
    # if block_K < 128, promote after 128/block_K iters.
    # if block_K > 128, promote after every iter.
    update_interval = 128 // block_K if block_K < 128 else 1

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                if (k + 1) % update_interval == 0:
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += C_local[i, j]
                    T.clear(C_local)
            # Tail processing
            if K_iters % update_interval != 0:
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j]
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def test_gemm_fp8(M, N, K, dtype):
    torch_dtype = map_torch_type(dtype)

    func = matmul(M, N, K, 128, 128, 64, dtype)

    kernel = tilelang.compile(func, out_idx=-1)

    a = torch.rand(M, K, dtype=torch.float16, device='cuda')
    a = (100 * (2 * a - 1)).to(dtype=torch_dtype)
    b = torch.rand(N, K, dtype=torch.float16, device='cuda')
    b = (100 * (2 * b - 1)).to(dtype=torch_dtype)

    c = kernel(a, b)

    ref_c = (a.float() @ b.float().T)

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


if __name__ == "__main__":
    test_gemm_fp8(1024, 1024, 8192, 'e4m3_float8')
    test_gemm_fp8(1024, 1024, 8192, 'e5m2_float8')
