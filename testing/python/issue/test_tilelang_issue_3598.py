"""Regression test for GitHub issue #3598.

A pipelined shared buffer accessed through a T.reshape / T.view must be
multi-versioned and waited for like the buffer itself.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

M, N, K, block_K = 64, 64, 256, 32
num_tiles = K // block_K


def _compile(func, out_idx):
    # Keep the cp.async software pipeline on targets with TMA as well.
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    return tilelang.compile(func, out_idx=out_idx, target="cuda", pass_configs=pass_configs)


def _randint(*shape):
    # Small integers keep the float16 GEMM exact.
    return torch.randint(-2, 3, shape, dtype=torch.float16, device="cuda")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipelined_gemm_copy_through_flat_view(num_stages):
    # The issue's kernel: each K-tile of A is contiguous and copied into a 1-D
    # view of the shared tile that the GEMM reads.
    @T.prim_func
    def main(
        A_tiles: T.Tensor((num_tiles, M * block_K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, block_K), T.float16)
            A_flat = T.reshape(A_shared, (M * block_K,))
            B_shared = T.alloc_shared((block_K, N), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(num_tiles, num_stages=num_stages):
                T.copy(A_tiles[k, :], A_flat)
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    kernel = _compile(main, out_idx=[2])
    a, b = _randint(M, K), _randint(K, N)
    a_tiles = a.view(M, num_tiles, block_K).permute(1, 0, 2).reshape(num_tiles, M * block_K)
    torch.testing.assert_close(kernel(a_tiles, b), a.float() @ b.float(), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("view_use", ["copy", "gemm", "both"])
def test_pipelined_gemm_through_same_shape_view(view_use):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, block_K), T.float16)
            A_view = T.view(A_shared, (M, block_K))
            B_shared = T.alloc_shared((block_K, N), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(num_tiles, num_stages=3):
                if view_use == "gemm":
                    T.copy(A[0, k * block_K], A_shared)
                else:
                    T.copy(A[0, k * block_K], A_view)
                T.copy(B[k * block_K, 0], B_shared)
                if view_use == "copy":
                    T.gemm(A_shared, B_shared, C_local)
                else:
                    T.gemm(A_view, B_shared, C_local)
            T.copy(C_local, C)

    kernel = _compile(main, out_idx=[2])
    a, b = _randint(M, K), _randint(K, N)
    torch.testing.assert_close(kernel(a, b), a.float() @ b.float(), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_pipelined_async_copy_through_view_is_waited_for():
    # The copy through the view is the only async copy, so the consumer reading
    # the buffer itself must wait for it.
    @T.prim_func
    def main(
        A_tiles: T.Tensor((num_tiles, M * block_K), T.float16),
        C: T.Tensor((M, block_K), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, block_K), T.float16)
            A_flat = T.reshape(A_shared, (M * block_K,))
            C_local = T.alloc_fragment((M, block_K), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(num_tiles, num_stages=2):
                T.copy(A_tiles[k, :], A_flat)
                for i, j in T.Parallel(M, block_K):
                    C_local[i, j] += A_shared[i, j]
            T.copy(C_local, C)

    kernel = _compile(main, out_idx=[1])
    a_tiles = _randint(num_tiles, M * block_K)
    ref = a_tiles.float().sum(0).view(M, block_K)
    torch.testing.assert_close(kernel(a_tiles), ref, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
