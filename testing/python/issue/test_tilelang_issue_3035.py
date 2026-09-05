"""Regression tests for T.gemm k_pack validation (issue #3035)."""

import pytest

import tilelang.language as T
import tilelang.testing


def _make_gemm(k_pack):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), "float16")
            B_shared = T.alloc_shared((128, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float16")
            T.gemm(A_shared, B_shared, C_local, k_pack=k_pack)

    return main


@pytest.mark.parametrize("k_pack", [1, 2])
def test_gemm_accepts_supported_k_pack(k_pack):
    assert _make_gemm(k_pack) is not None


@pytest.mark.parametrize("k_pack", [0, -1, 3, 4, 8, 16])
def test_gemm_rejects_unsupported_k_pack(k_pack):
    with pytest.raises(ValueError, match=rf"T\.gemm k_pack must be an int equal to 1 or 2, got {k_pack}"):
        _make_gemm(k_pack)


@pytest.mark.parametrize("k_pack", [True, 2.0, "2", None])
def test_gemm_rejects_non_integer_k_pack(k_pack):
    with pytest.raises(ValueError) as exc_info:
        _make_gemm(k_pack)
    assert str(exc_info.value) == f"T.gemm k_pack must be an int equal to 1 or 2, got {k_pack!r}"


if __name__ == "__main__":
    tilelang.testing.main()
