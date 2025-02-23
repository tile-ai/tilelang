# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j, k in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def assert_gemm_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float",
):
    func = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    print(func)

    rt_mod, _ = tilelang.lower(func, target="webgpu")

    src_code = rt_mod.imported_modules[0].get_source()

    assert src_code is not None


def test_gemm_codegen():
    assert_gemm_codegen(1024, 1024, 1024, 16, 16, 16)


if __name__ == "__main__":
    tilelang.testing.main()
