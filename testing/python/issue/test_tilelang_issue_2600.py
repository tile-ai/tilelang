"""Regression tests for T.dp4a dtype validation (issue #2600)."""

import pytest

import tilelang.language as T
import tilelang.testing


def _build_dp4a(a_dtype: str, b_dtype: str, c_dtype: str):
    @T.prim_func
    def kernel(
        A: T.Tensor((4,), a_dtype),
        B: T.Tensor((4,), b_dtype),
        C: T.Tensor((1,), c_dtype),
    ):
        T.dp4a(A, B, C)

    return kernel


def test_dp4a_accepts_int8_int32():
    kernel = _build_dp4a("int8", "int8", "int32")
    assert kernel is not None


def test_dp4a_accepts_buffer_load_slice():
    # Existing SIMT kernels call dp4a with BufferLoad arguments.
    @T.prim_func
    def kernel(
        A: T.Tensor((4,), "int8"),
        B: T.Tensor((4,), "int8"),
        C: T.Tensor((1,), "int32"),
    ):
        T.dp4a(A[0], B[0], C[0])

    assert kernel is not None


@pytest.mark.parametrize(
    "a_dtype,b_dtype,c_dtype",
    [
        ("float16", "float16", "int32"),  # wrong input dtype
        ("int8", "int8", "float32"),  # wrong accumulator dtype
        ("uint8", "uint8", "int32"),  # unsigned inputs (signed-only intrinsic)
        ("int8", "float16", "int32"),  # mixed input dtypes
    ],
)
def test_dp4a_rejects_invalid_dtypes(a_dtype, b_dtype, c_dtype):
    with pytest.raises(ValueError, match="dp4a requires"):
        _build_dp4a(a_dtype, b_dtype, c_dtype)


if __name__ == "__main__":
    tilelang.testing.main()
