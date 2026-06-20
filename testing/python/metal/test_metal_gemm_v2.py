"""Test Metal gemm_v2 with actual execution on Metal hardware.

These tests verify correctness of T.gemm (gemm_v2) using simdgroup matrix
operations by comparing results against torch.matmul.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch


@tilelang.jit
def matmul_gemm_v2(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


def assert_gemm_v2(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
    atol=1e-2,
):
    jit_kernel = matmul_gemm_v2(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = dtype.as_torch()
    torch_accum_dtype = accum_dtype.as_torch()
    a = torch.randn(M, K, dtype=torch_dtype, device="mps")
    b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_accum_dtype, device="mps")

    jit_kernel(a, b, c)

    ref = a.to(torch_accum_dtype) @ b.to(torch_accum_dtype)
    assert torch.allclose(ref, c, atol=atol), (
        f"Result mismatch for M={M}, N={N}, K={K}, "
        f"block=({block_M},{block_N},{block_K}), dtype={dtype}\n"
        f"max diff: {(ref - c).abs().max().item()}"
    )


@tilelang.jit
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
    def gemm_kernel(
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

    return gemm_kernel


def assert_gemm_v2_global_c(
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
    atol=1e-2,
    rtol=1e-2,
):
    jit_kernel = matmul_gemm_v2_global_c(
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

    torch_dtype = dtype.as_torch()
    torch_accum_dtype = accum_dtype.as_torch()
    a = torch.randn(M, K, dtype=torch_dtype, device="mps")
    b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_accum_dtype, device="mps")

    jit_kernel(a, b, c)

    if accum_dtype == T.float16:
        ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch_accum_dtype)
    else:
        ref = a.to(torch_accum_dtype) @ b.to(torch_accum_dtype)
    assert torch.allclose(ref, c, atol=atol, rtol=rtol), (
        f"Result mismatch for direct global C, M={M}, N={N}, K={K}, "
        f"block=({block_M},{block_N}), dtype={dtype}\n"
        f"max diff: {(ref - c).abs().max().item()}"
    )


@tilelang.testing.requires_metal
def test_gemm_v2_16x16x16():
    assert_gemm_v2(128, 128, 128, 16, 16, 16)


@tilelang.testing.requires_metal
def test_gemm_v2_16x16x8():
    assert_gemm_v2(128, 128, 128, 16, 16, 8)


@tilelang.testing.requires_metal
def test_gemm_v2_large():
    assert_gemm_v2(128, 128, 128, 32, 32, 32)


@tilelang.testing.requires_metal
def test_gemm_v2_cooperative_tensor_non_square():
    assert_gemm_v2(128, 128, 128, 32, 64, 32)


@tilelang.testing.requires_metal
def test_gemm_v2_cooperative_tensor_global_c():
    assert_gemm_v2_global_c(256, 256, 256, 64, 128)


@tilelang.testing.requires_metal
def test_gemm_v2_cooperative_tensor_global_c_fp16_mlx_swizzle():
    assert_gemm_v2_global_c(
        256,
        256,
        256,
        64,
        128,
        accum_dtype=T.float16,
        swizzle_panel=4,
        swizzle_order="mlx",
        atol=1e-1,
        rtol=1e-2,
    )


@tilelang.testing.requires_metal
def test_gemm_v2_cooperative_tensor_global_c_fp16_mlx_swizzle_multirow():
    assert_gemm_v2_global_c(
        512,
        256,
        256,
        64,
        128,
        accum_dtype=T.float16,
        swizzle_panel=4,
        swizzle_order="mlx",
        atol=1e-1,
        rtol=1e-2,
    )


@tilelang.testing.requires_metal
def test_gemm_v2_1024():
    assert_gemm_v2(1024, 1024, 1024, 16, 16, 16, atol=1.0)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
