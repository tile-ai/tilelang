import pytest

import tilelang.language as T
from tvm import tirx


FASTMATH_INTRINSICS = [
    T.__exp,
    T.__exp10,
    T.__log,
    T.__log2,
    T.__log10,
    T.__tan,
    T.__cos,
    T.__sin,
]


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int16",
        "int32",
        "uint16",
        "uint32",
        "float8_e4m3fn",
        "float8_e5m2",
    ],
)
@pytest.mark.parametrize("intrinsic", FASTMATH_INTRINSICS)
def test_fastmath_rejects_unsupported_dtype(intrinsic, dtype):
    value = tirx.Var("value", dtype)

    with pytest.raises(TypeError, match=rf"T\.{intrinsic.__name__} only supports floating-point inputs"):
        intrinsic(value)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("intrinsic", FASTMATH_INTRINSICS)
def test_fastmath_accepts_float_dtype_at_frontend(intrinsic, dtype):
    value = tirx.Var("value", dtype)
    result = intrinsic(value)

    assert isinstance(result, tirx.Call)
    assert result.dtype == dtype
