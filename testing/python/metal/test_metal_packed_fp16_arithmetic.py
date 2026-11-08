import pytest
from tilelang import tvm

import tilelang
import tilelang.language as T
import tilelang.testing


def _lower_vectorized_binary(op, lanes):
    @T.prim_func
    def kernel(
        a: T.Tensor((lanes,), T.float16),
        b: T.Tensor((lanes,), T.float16),
        c: T.Tensor((lanes,), T.float16),
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
    ("op", "expected"),
    [
        (lambda a, b: a + b, ") + as_type<half2>"),
        (lambda a, b: a - b, ") - as_type<half2>"),
        (lambda a, b: a * b, ") * as_type<half2>"),
        (lambda a, b: a / b, ") / as_type<half2>"),
        (T.min, "min(as_type<half2>"),
        (T.max, "max(as_type<half2>"),
    ],
)
def test_packed_float16_binary_operations_use_half2(op, expected):
    source = _lower_vectorized_binary(op, 8)

    assert "uint4" in source
    assert "as_type<half2>" in source
    assert expected in source


def test_native_float16_vector_arithmetic_is_unchanged():
    source = _lower_vectorized_binary(lambda a, b: a + b, 4)

    assert "half4" in source
    assert "as_type<half2>" not in source


if __name__ == "__main__":
    tilelang.testing.main()
