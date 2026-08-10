"""Regression test for GitHub issue #2928.

A ``block_N`` that is itself a legal Hopper WGMMA width (96, 112, 160, ...)
must lower to a single ``m64nNk16`` instruction.  Deriving the instruction
width from ``gcd(warp_col_tiles, 256)`` decomposed those extents into n32 / n16
atoms and gave up most of the tensor-core throughput.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import select_wgmma_inst_n


@pytest.mark.parametrize(
    "warp_col_tiles, expected",
    [
        # Legal single-instruction extents: keep the whole width.
        (64, 64),
        (96, 96),
        (112, 112),
        (160, 160),
        (176, 176),
        (256, 256),
        # ``N % 16 == 8`` is a separately broken class (issue #2593); the width
        # selection is left exactly as it was.
        (24, 8),
        (40, 8),
        (88, 8),
        # Past the 256 ceiling: widest legal divisor, which gcd cannot reach.
        (384, 192),
        (512, 256),
    ],
)
def test_select_wgmma_inst_n(warp_col_tiles, expected):
    assert select_wgmma_inst_n(warp_col_tiles) == expected


def _matmul_nt(M, N, K, block_M, block_N, block_K):
    dtype, accum_dtype = "float16", "float"

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
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _emitted_shapes(kernel, op):
    """Distinct ``m, n, k`` operands emitted for ``op`` (``wgmma_ss``, ``wgmma_sp_ss``)."""
    return sorted(set(re.findall(rf"{op}<[^>]*?, (\d+, \d+, \d+),", kernel.get_kernel_source())))


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize("block_N", [96, 112, 160])
def test_gemm_emits_single_wide_wgmma(block_N):
    M, K, block_M, block_K = 128, 128, 64, 64
    N = block_N * 2

    kernel = tilelang.compile(_matmul_nt(M, N, K, block_M, block_N, block_K))
    assert _emitted_shapes(kernel, "wgmma_ss") == [f"64, {block_N}, 16"]

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float16, device="cuda")
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    kernel(a, b, c)
    torch.testing.assert_close(c, (a.float() @ b.float().T).half(), rtol=1e-2, atol=1e-2)


def _matmul_sp_nt(M, N, K, block_M, block_N, block_K, e_factor):
    dtype, accum_dtype, meta_dtype = "float16", "float", "int16"

    @T.prim_func
    def main(
        A_sparse: T.Tensor((M, K // 2), dtype),
        E: T.Tensor((M, K // e_factor), meta_dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), dtype)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), meta_dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, False, False, True)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize("block_N", [96, 112, 160])
def test_gemm_sp_emits_single_wide_wgmma(block_N):
    # Imported inside the sm_90 gate so collection never depends on examples/.
    from examples.gemm_sp.sparse_utils import compress, get_e_factor, randn_semi_sparse

    M, K, block_M, block_K = 128, 32, 128, 32
    N = block_N * 2
    e_factor = get_e_factor(T.float16, T.int16)

    kernel = tilelang.compile(
        _matmul_sp_nt(M, N, K, block_M, block_N, block_K, e_factor),
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    # Sparse WGMMA doubles K per instruction: 512 bits / 16 bits per element.
    assert _emitted_shapes(kernel, "wgmma_sp_ss") == [f"64, {block_N}, 32"]

    torch.manual_seed(0)
    a = randn_semi_sparse(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float16, device="cuda")
    a_sparse, e = compress(a, meta_dtype=torch.int16)
    c = kernel(a_sparse, e, b)
    torch.testing.assert_close(c.float(), a.float() @ b.float().T, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
