import pytest

import tilelang.language as T
import tilelang.testing


def _make_gemm(M, N, K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), "float16")
            B_shared = T.alloc_shared((K, N), "float16")
            C_local = T.alloc_fragment((M, N), "float16")
            T.gemm(A_shared, B_shared, C_local)

    return main


def _make_gemm_sp(M, N, Kc):
    @T.prim_func
    def main(
        A_sparse: T.Tensor((M, Kc), "float16"),
        E: T.Tensor((M, Kc // 4), "uint8"),
        B: T.Tensor((2 * Kc, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, Kc), "float16")
            E_shared = T.alloc_shared((M, Kc // 4), "uint8")
            B_shared = T.alloc_shared((2 * Kc, N), "float16")
            C_local = T.alloc_fragment((M, N), "float16")
            T.gemm_sp(A_shared, E_shared, B_shared, C_local)

    return main


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [(T.dynamic("M"), 128, 32), (128, T.dynamic("N"), 32), (128, 128, T.dynamic("K"))],
)
def test_gemm_rejects_symbolic_dimension(M, N, K):
    with pytest.raises(ValueError):
        _make_gemm(M, N, K)


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [(T.dynamic("M"), 128, 32), (128, T.dynamic("N"), 32), (128, 128, T.dynamic("K"))],
)
def test_gemm_sp_rejects_symbolic_dimension(M, N, K):
    with pytest.raises(ValueError):
        _make_gemm_sp(M, N, K)


if __name__ == "__main__":
    tilelang.testing.main()
