"""Regression for per-tile K tails in the sparse MMA fallback."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.intrinsics.sparse_layout import get_e_factor


def _make_sparse_gemm(K: int, block_K: int):
    M = N = 128
    meta_dtype = "int16"
    e_factor = get_e_factor(T.int8, T.dtype(meta_dtype))

    @T.prim_func
    def main(
        A: T.Tensor((M, K // 2), "int8"),
        E: T.Tensor((M, K // e_factor), meta_dtype),
        B: T.Tensor((N, K), "int8"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(1, 1, threads=128):
            A_shared = T.alloc_shared((M, block_K // 2), "int8")
            E_shared = T.alloc_shared((M, block_K // e_factor), meta_dtype)
            B_shared = T.alloc_shared((N, block_K), "int8")
            C_local = T.alloc_fragment((M, N), "int32")
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[0, k * block_K // 2], A_shared)
                T.copy(E[0, k * block_K // e_factor], E_shared)
                T.copy(B[0, k * block_K], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0, 0])

    return main


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_gemm_sp_rejects_per_tile_k_tail():
    # The total K is a multiple of the 64-element int8 MMA atom, but each
    # 96-element tile would independently drop its final 32 elements.
    with pytest.raises(ValueError, match="K tile size 96.*divisible.*64"):
        tilelang.compile(
            _make_sparse_gemm(K=192, block_K=96),
            target="cuda",
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        )
