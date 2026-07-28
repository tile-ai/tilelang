import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import tirx


def test_shuffle_is_reexported_from_tir_ir():
    from tilelang.language.tir import ir as tir_ir

    assert tir_ir.Shuffle is tirx.Shuffle
    assert T.Shuffle is tirx.Shuffle
    assert "Shuffle" in tir_ir.__all__


@pytest.mark.parametrize(
    "dtype,lane,expected_dtype",
    [
        ("bfloat16x2", 0, "bfloat16"),
        ("float16x2", 1, "float16"),
        ("int8x4", 2, "int8"),
    ],
)
def test_extract_lane_builds_shuffle(dtype, lane, expected_dtype):
    vector = tirx.Var("vector", dtype)
    result = T.extract_lane(vector, lane)

    assert isinstance(result, tirx.Shuffle)
    assert str(result.dtype) == expected_dtype
    assert result.vectors[0].same_as(vector)
    assert result.indices[0].value == lane


def test_extract_lane_accepts_int_imm():
    vector = tirx.Var("vector", "bfloat16x2")
    result = T.extract_lane(vector, tirx.IntImm("int32", 1))

    assert result.indices[0].value == 1


def test_extract_lane_validates_inputs():
    vector = tirx.Var("vector", "bfloat16x2")

    with pytest.raises(TypeError, match="expects a PrimExpr"):
        T.extract_lane(1, 0)
    with pytest.raises(ValueError, match="expects a vector expression"):
        T.extract_lane(tirx.Var("scalar", "bfloat16"), 0)
    with pytest.raises(TypeError, match="compile-time integer"):
        T.extract_lane(vector, tirx.Var("lane", "int32"))
    with pytest.raises(IndexError, match="out of bounds"):
        T.extract_lane(vector, -1)
    with pytest.raises(IndexError, match="out of bounds"):
        T.extract_lane(vector, 2)


def _make_extract_lane_kernel(dtype, vector_dtype):
    @T.prim_func
    def kernel(A: T.Tensor((2,), dtype), B: T.Tensor((2,), dtype)):
        with T.Kernel(1, threads=1):
            packed = T.alloc_var(vector_dtype)
            packed = vector_dtype(A[0], A[1])
            B[0] = T.extract_lane(packed, 0)
            B[1] = T.extract_lane(packed, 1)

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype,vector_dtype,scalar_type",
    [
        (T.bfloat16, T.bfloat16x2, "bfloat16_t"),
        (T.float16, T.float16x2, "half_t"),
    ],
)
def test_extract_lane_cuda(dtype, vector_dtype, scalar_type):
    kernel = tilelang.compile(_make_extract_lane_kernel(dtype, vector_dtype), out_idx=-1, target="cuda")
    source = kernel.get_kernel_source()
    assert f"B[0] = {scalar_type}(" in source
    assert f"B[1] = {scalar_type}(" in source

    input_tensor = torch.tensor([1.25, -2.5], dtype=dtype.as_torch(), device="cuda")
    output = kernel(input_tensor)
    torch.testing.assert_close(output, input_tensor)


if __name__ == "__main__":
    tilelang.testing.main()
