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


def matmul_gemm_v2_shared_c(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared")

            T.clear(C_shared)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def matmul_gemm_v2_global_c(
    M,
    N,
    K,
    block_M,
    block_N,
    dtype=T.float16,
    accum_dtype=T.float32,
    threads=128,
    swizzle_panel=0,
    swizzle_order="row",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        tiles_n = T.ceildiv(N, block_N)
        tiles_m = T.ceildiv(M, block_M)
        use_mlx_swizzle = swizzle_panel and swizzle_order == "mlx"
        grid_n = tiles_n * swizzle_panel if use_mlx_swizzle else tiles_n
        grid_m = T.ceildiv(tiles_m, swizzle_panel) if use_mlx_swizzle else tiles_m
        with T.Kernel(grid_n, grid_m, threads=threads) as (bx, by):
            logical_bx = bx // swizzle_panel if use_mlx_swizzle else bx
            logical_by = by * swizzle_panel + bx % swizzle_panel if use_mlx_swizzle else by

            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel, order=swizzle_order)
            if use_mlx_swizzle:
                if logical_by < tiles_m:
                    T.gemm(
                        A[logical_by * block_M : (logical_by + 1) * block_M, 0:K],
                        B[0:K, logical_bx * block_N : (logical_bx + 1) * block_N],
                        C[
                            logical_by * block_M : (logical_by + 1) * block_M,
                            logical_bx * block_N : (logical_bx + 1) * block_N,
                        ],
                        clear_accum=True,
                    )
            else:
                T.gemm(
                    A[logical_by * block_M : (logical_by + 1) * block_M, 0:K],
                    B[0:K, logical_bx * block_N : (logical_bx + 1) * block_N],
                    C[
                        logical_by * block_M : (logical_by + 1) * block_M,
                        logical_bx * block_N : (logical_bx + 1) * block_N,
                    ],
                    clear_accum=True,
                )

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
    assert "main_kernel" in src_code
    # Verify simdgroup matrix operations are present
    assert "simdgroup_multiply_accumulate" in src_code
    assert "simdgroup_load" in src_code
    assert "simdgroup_store" in src_code


def assert_metal_gemm_v2_cooperative_tensor_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    func = matmul_gemm_v2_shared_c(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "main_kernel" in src_code
    assert "mpp::tensor_ops::matmul2d" in src_code
    assert "cooperative_tensor" in src_code


def assert_metal_gemm_v2_global_cooperative_tensor_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    dtype=T.float16,
    accum_dtype=T.float32,
    threads=128,
    swizzle_panel=0,
    swizzle_order="row",
):
    func = matmul_gemm_v2_global_c(
        M,
        N,
        K,
        block_M,
        block_N,
        dtype=dtype,
        accum_dtype=accum_dtype,
        threads=threads,
        swizzle_panel=swizzle_panel,
        swizzle_order=swizzle_order,
    )
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "main_kernel" in src_code
    assert "mpp::tensor_ops::matmul2d" in src_code
    assert "const device half* __restrict A" in src_code
    assert "const device half* __restrict B" in src_code
    assert "const device half* __src" in src_code
    assert "[[simdgroup_index_in_threadgroup]]" in src_code
    assert "__metal_get_thread_index_in_simdgroup" in src_code
    assert "max_total_threads_per_threadgroup(128)" in src_code
    assert "threadgroup half" not in src_code
    assert "thread float C_ct" not in src_code
    assert "blockIdx.x" in src_code
    if swizzle_order != "mlx":
        assert "blockIdx.y" in src_code
    if accum_dtype == T.float16:
        assert "device half* __restrict C" in src_code
        assert "half4(*(thread float4*)" in src_code
    if swizzle_order == "mlx":
        assert "__physical_blockIdx" in src_code
        assert "__physical_blockIdx.x >> 2" in src_code
        assert "__physical_blockIdx.x & 3u" in src_code
        assert "blockIdx.x) >> 2" not in src_code
        assert "blockIdx.x) & 3" not in src_code
        if (M + block_M - 1) // block_M > swizzle_panel:
            assert f"* {block_M * K * swizzle_panel}" not in src_code


def test_metal_gemm_v2_float16():
    assert_metal_gemm_v2_codegen(64, 64, 64, 16, 16, 16, dtype=T.float16)


def test_metal_gemm_v2_float32():
    assert_metal_gemm_v2_codegen(64, 64, 64, 16, 16, 16, dtype=T.float32, accum_dtype=T.float32)


def test_metal_gemm_v2_larger():
    assert_metal_gemm_v2_codegen(128, 128, 128, 32, 32, 32, dtype=T.float16)


def test_metal_gemm_v2_cooperative_tensor_codegen():
    assert_metal_gemm_v2_cooperative_tensor_codegen(128, 128, 128, 32, 64, 32, dtype=T.float16)


def test_metal_gemm_v2_global_cooperative_tensor_codegen():
    assert_metal_gemm_v2_global_cooperative_tensor_codegen(128, 256, 128, 64, 128, dtype=T.float16)


def test_metal_gemm_v2_global_cooperative_tensor_mlx_swizzle_codegen():
    assert_metal_gemm_v2_global_cooperative_tensor_codegen(
        512,
        256,
        128,
        64,
        128,
        dtype=T.float16,
        accum_dtype=T.float16,
        swizzle_panel=4,
        swizzle_order="mlx",
    )


def test_metal_gemm_v2_small_blocks():
    """Test with blocks where warp_rows > 1 and warp_cols > 1, which previously
    produced incorrect results due to swizzle padding changing the stride.
    """
    assert_metal_gemm_v2_codegen(16, 16, 16, 16, 16, 16, dtype=T.float16)


if __name__ == "__main__":
    tilelang.testing.main()
