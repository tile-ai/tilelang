"""Test Metal gemm_v2 code generation on any platform (including Linux).

These tests verify that TileLang can compile kernels using T.gemm (gemm_v2)
down to Metal shader source code with simdgroup matrix operations,
without requiring a Metal runtime or macOS.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul_gemm_v2(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def assert_metal_gemm_v2_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    func = matmul_gemm_v2(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "kernel void" in src_code
    # Verify simdgroup matrix operations are present
    assert "simdgroup_multiply_accumulate" in src_code
    assert "simdgroup_load" in src_code
    assert "simdgroup_store" in src_code


def test_metal_gemm_v2_float16():
    assert_metal_gemm_v2_codegen(64, 64, 64, 16, 16, 16, dtype=T.float16)


def test_metal_gemm_v2_float32():
    assert_metal_gemm_v2_codegen(64, 64, 64, 16, 16, 16, dtype=T.float32, accum_dtype=T.float32)


def test_metal_gemm_v2_larger():
    assert_metal_gemm_v2_codegen(128, 128, 128, 32, 32, 32, dtype=T.float16)


def test_metal_gemm_v2_small_blocks():
    """Test with blocks where warp_rows > 1 and warp_cols > 1, which previously
    produced incorrect results due to swizzle padding changing the stride.
    """
    assert_metal_gemm_v2_codegen(16, 16, 16, 16, 16, 16, dtype=T.float16)


if __name__ == "__main__":
    tilelang.testing.main()
