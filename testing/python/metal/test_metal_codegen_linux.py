"""Test Metal code generation on any platform (including Linux).

These tests verify that TileLang can compile kernels down to Metal shader
source code without requiring a Metal runtime or macOS.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float32, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def assert_metal_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
):
    func = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "metal" in src_code or "kernel void" in src_code


def test_metal_codegen_float32():
    assert_metal_codegen(1024, 1024, 1024, 16, 16, 16)


def test_metal_codegen_float16():
    assert_metal_codegen(1024, 1024, 1024, 16, 16, 16, dtype=T.float16)


def test_metal_codegen_int32():
    assert_metal_codegen(1024, 1024, 1024, 16, 16, 16, dtype=T.int32)


if __name__ == "__main__":
    tilelang.testing.main()
