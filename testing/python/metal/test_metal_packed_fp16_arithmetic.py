import pytest
import torch
from tilelang import tvm

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_vectorized_binary(op, lanes, dtype=T.float16):
    @T.prim_func
    def kernel(
        a: T.Tensor((lanes,), dtype),
        b: T.Tensor((lanes,), dtype),
        c: T.Tensor((lanes,), dtype),
    ):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(lanes):
                c[i] = op(a[i], b[i])

    return kernel


def _lower_vectorized_binary(op, lanes):
    kernel = _make_vectorized_binary(op, lanes)
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
def test_packed_binary_operations_use_native_component_arithmetic(op, op_name):
    source = _lower_vectorized_binary(op, 8)

    assert "uint4" in source
    assert source.count("(") == source.count(")")
    assert f"__tilelang_packed_half2_binary(a[0], b[0], __tilelang_packed_half2_{op_name}())" in source


@pytest.mark.parametrize("lanes", [6, 8])
@pytest.mark.parametrize(("op", "op_name"), [(T.min, "min"), (T.max, "max")])
def test_packed_min_max_accept_two_arguments(lanes, op, op_name):
    source = _lower_vectorized_binary(op, lanes)

    assert f"TILELANG_PACKED_BINARY_FUNCTOR({op_name}, {op_name}(a, b))" in source
    assert f"{op_name}(a b)" not in source
    if lanes == 8:
        assert f"__tilelang_packed_half2_{op_name}()" in source


def test_native_float16_vector_arithmetic_is_unchanged():
    source = _lower_vectorized_binary(lambda a, b: a + b, 4)

    assert "half4" in source
    assert "c[0] = (a[0] + b[0]);" in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize(("dtype", "lanes"), [(T.float32, 1), (T.float16, 4), (T.float16, 8)])
def test_metal_compiles_and_runs_vectorized_binary(dtype, lanes):
    # Exercise a composed expression to check the helper's by-value operands.
    kernel = _make_vectorized_binary(lambda a, b: (a + b) * b, lanes, dtype)
    compiled = tilelang.compile(kernel, out_idx=[], target="metal")
    a = torch.arange(lanes, dtype=dtype.as_torch(), device="mps")
    b = torch.full_like(a, 2)
    c = torch.empty_like(a)
    compiled(a, b, c)
    torch.testing.assert_close(c, (a + b) * b, rtol=0, atol=0)


@tilelang.testing.requires_metal
def test_metal_packed_float16_in_place_add_twice():
    @T.prim_func
    def kernel(a: T.Tensor((8,), T.float16), b: T.Tensor((8,), T.float16)):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(8):
                a[i] = a[i] + b[i]
            for i in T.vectorized(8):
                a[i] = a[i] + b[i]

    compiled = tilelang.compile(kernel, out_idx=[], target="metal")
    a = torch.ones(8, dtype=torch.float16, device="mps")
    b = torch.ones_like(a)
    compiled(a, b)
    torch.testing.assert_close(a, torch.full_like(a, 3), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
