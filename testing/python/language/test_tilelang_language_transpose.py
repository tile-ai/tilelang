"""Tests for T.transpose shared memory transpose primitive."""

import tilelang
import tilelang.language as T
import torch


def tilelang_transpose(M, N, block_M, block_N, dtype=T.float16):
    """Kernel: read tile from A into shared, transpose in shared, write to B.

    A is (M, N), B is (M, N).
    B = A.T.T = A when block_M == M and block_N == N (single tile).
    Actually: we read A tile (block_M, block_N) into shared,
    transpose to (block_N, block_M) in shared, then write to B
    so B[bx*block_N + j, by*block_M + i] = A[by*block_M + i, bx*block_N + j]
    i.e., B = A.T
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            tile = T.alloc_shared((block_M, block_N), dtype)
            tile_T = T.alloc_shared((block_N, block_M), dtype)

            # Load from global to shared
            T.copy(
                A[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                tile,
            )
            # Transpose in shared memory
            T.transpose(tile, tile_T)
            # Store transposed tile back to global
            T.copy(
                tile_T,
                B[bx * block_N : (bx + 1) * block_N, by * block_M : (by + 1) * block_M],
            )

    return main


def run_tilelang_transpose(M=128, N=128, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_transpose(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    expected = a.T
    torch.testing.assert_close(b, expected, rtol=1e-2, atol=1e-2)
    print(f"PASS: transpose M={M}, N={N}, block_M={block_M}, block_N={block_N}")


def tilelang_transpose_square(M, block_M, dtype=T.float16):
    """Simpler test: square transpose with single tile."""

    @T.prim_func
    def main(
        A: T.Tensor((M, M), dtype),
        B: T.Tensor((M, M), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(M, block_M), threads=128) as (bx, by):
            tile = T.alloc_shared((block_M, block_M), dtype)
            tile_T = T.alloc_shared((block_M, block_M), dtype)

            T.copy(
                A[by * block_M : (by + 1) * block_M, bx * block_M : (bx + 1) * block_M],
                tile,
            )
            T.transpose(tile, tile_T)
            T.copy(
                tile_T,
                B[bx * block_M : (bx + 1) * block_M, by * block_M : (by + 1) * block_M],
            )

    return main


def run_tilelang_transpose_square(M=256, block_M=128, dtype=T.float16):
    program = tilelang_transpose_square(M, block_M, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, M, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    expected = a.T
    torch.testing.assert_close(b, expected, rtol=1e-2, atol=1e-2)
    print(f"PASS: square transpose M={M}, block_M={block_M}")


@tilelang.testing.requires_cuda
def test_tilelang_transpose():
    run_tilelang_transpose(M=128, N=128, block_M=128, block_N=128)
    run_tilelang_transpose(M=256, N=256, block_M=128, block_N=128)
    run_tilelang_transpose(M=128, N=256, block_M=128, block_N=256)


@tilelang.testing.requires_cuda
def test_tilelang_transpose_square():
    run_tilelang_transpose_square(M=128, block_M=128)
    run_tilelang_transpose_square(M=256, block_M=128)
    run_tilelang_transpose_square(M=512, block_M=128)


if __name__ == "__main__":
    tilelang.testing.main()
