"""Regression test for GitHub issue #2522."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.language import GemmWarpPolicy


def _matmul(block_M, block_N, block_K, threads, policy, pass_configs=None):
    """Single-block GEMM: C[block_M, block_N] = A @ B over K = 2 * block_K."""
    M, N, K = block_M, block_N, block_K * 2

    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_frag = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_frag, policy=policy)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return tilelang.compile(main, out_idx=-1, pass_configs=pass_configs)


def _check(kernel, M, N, K, seed=0):
    torch.manual_seed(seed)
    A = torch.randint(0, 3, (M, K), dtype=torch.float16, device="cuda")
    B = torch.randint(0, 3, (K, N), dtype=torch.float16, device="cuda")
    ref = A.float() @ B.float()
    out = kernel(A, B).float()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_fullrow_two_warpgroups_along_m():
    """FullRow, block_M=256, threads=256: m_warp=8 (two warpgroups along M).

    The configuration reported in issue #2522: without the warpgroup-major
    store layout, output rows [64:128) and [128:192) are exchanged.
    """
    kernel = _matmul(256, 128, 32, 256, GemmWarpPolicy.FullRow)
    assert "wgmma" in kernel.get_kernel_source()
    _check(kernel, 256, 128, 64)


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_fullrow_single_warpgroup():
    """FullRow, block_M=128, threads=128: single warpgroup along M."""
    kernel = _matmul(128, 128, 32, 128, GemmWarpPolicy.FullRow)
    assert "wgmma" in kernel.get_kernel_source()
    _check(kernel, 128, 128, 64)


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_two_warpgroups_with_col_warps():
    """FullRow, block_M=128, threads=512: m_warp=8, n_warp=2.

    Exercises block_col_warps > 1 together with multiple warpgroups along M
    (warp_rows=1).  Guards the ordering of the inter-warpgroup and warp_n
    thread repeats: warp_id = warp_m + block_row_warps * warp_n, so warp_n
    must be the outermost repeat, not interleaved between the in-group warps
    and the warpgroup index.
    """
    kernel = _matmul(128, 256, 32, 512, GemmWarpPolicy.FullRow)
    assert "wgmma" in kernel.get_kernel_source()
    _check(kernel, 128, 256, 64)


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_two_warpgroups_col_warps_multi_atom():
    """Square, block_M=256, block_N=16, threads=512: m_warp=8, n_warp=2, warp_rows=2.

    All three layers of the M decomposition are active at once: multiple
    warpgroups along M, multiple warps along N, and multiple 64-row WGMMA
    atoms per warpgroup.  Uses K-major B (transpose_B) since block_N=16
    N-major shared layouts are not supported.
    """
    M, N, K = 256, 16, 64
    block_K, threads = 32, 512

    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((N, K), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1, 1, threads=threads) as (bx, by):
            A_shared = T.alloc_shared((M, block_K), "float16")
            B_shared = T.alloc_shared((N, block_K), "float16")
            C_frag = T.alloc_fragment((M, N), "float32")
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[0, k * block_K], A_shared)
                T.copy(B[0, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_frag, transpose_B=True, policy=GemmWarpPolicy.Square)
            T.copy(C_frag, C[0, 0])

    kernel = tilelang.compile(main, out_idx=-1)
    assert "wgmma" in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randint(0, 3, (M, K), dtype=torch.float16, device="cuda")
    B = torch.randint(0, 3, (N, K), dtype=torch.float16, device="cuda")
    ref = A.float() @ B.float().T
    out = kernel(A, B).float()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
