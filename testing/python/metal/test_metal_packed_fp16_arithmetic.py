import pytest
from tilelang import tvm

import tilelang
import tilelang.language as T
import tilelang.testing


def _lower_vectorized_binary(op, lanes, dtype=T.float16):
    @T.prim_func
    def kernel(
        a: T.Tensor((lanes,), dtype),
        b: T.Tensor((lanes,), dtype),
        c: T.Tensor((lanes,), dtype),
    ):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(lanes):
                c[i] = op(a[i], b[i])

    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(
            kernel,
            target="metal",
            enable_device_compile=False,
        )
    return artifact.kernel_source or ""


@pytest.mark.parametrize(
    ("op", "op_name"),
    [
        (lambda a, b: a + b, "add"),
        (lambda a, b: a - b, "sub"),
        (lambda a, b: a * b, "mul"),
        (lambda a, b: a / b, "div"),
        (T.min, "min"),
        (T.max, "max"),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "family"),
    [
        (T.float16, "half2"),
        (T.bfloat16, "bfloat2"),
    ],
)
def test_packed_binary_operations_use_native_component_arithmetic(op, op_name, dtype, family):
    source = _lower_vectorized_binary(op, 8, dtype)

    assert "uint4" in source
    assert source.count("(") == source.count(")")
    assert (f"__tilelang_packed_{family}_binary(a[0], b[0], __tilelang_packed_{family}_{op_name}())") in source


@pytest.mark.parametrize("lanes", [6, 8])
@pytest.mark.parametrize(("op", "op_name"), [(T.min, "min"), (T.max, "max")])
def test_packed_min_max_accept_two_arguments(lanes, op, op_name):
    source = _lower_vectorized_binary(op, lanes)

    assert f"return {op_name}(a, b);" in source
    assert "return min(a b);" not in source
    assert "return max(a b);" not in source


def test_native_float16_vector_arithmetic_is_unchanged():
    source = _lower_vectorized_binary(lambda a, b: a + b, 4)

    assert "half4" in source
    assert "c[0] = (a[0] + b[0]);" in source


if __name__ == "__main__":
    tilelang.testing.main()
