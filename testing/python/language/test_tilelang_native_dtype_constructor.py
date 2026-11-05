"""Preserve scalar broadcasting and lane conversion at the public dtype boundary."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as C
import tilelang.language as T


@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("value", [-2.75, 0.0, 7.5])
def test_scalar_to_vector_constructor_executes(dtype, value):
    vector = getattr(T, dtype + "x4")

    @C.prim_func
    def kernel(output: C.Tensor((4,), dtype), scalar: C.float32):
        output[C.Ramp(0, 1, 4)] = vector(scalar)

    target = {"kind": "c"}
    executable = tilelang.compile(kernel, target=target, target_host=target, execution_backend="tvm_ffi")
    actual = torch.empty((4,), dtype=getattr(torch, dtype))
    executable(actual, value)
    reference = int(value) if dtype == "int32" else value
    torch.testing.assert_close(actual, torch.full_like(actual, reference), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_float_literal_to_vector_constructor_executes(dtype):
    vector = getattr(T, dtype + "x4")

    @C.prim_func
    def kernel(output: C.Tensor((4,), dtype)):
        output[C.Ramp(0, 1, 4)] = vector(2.75)

    target = {"kind": "c"}
    executable = tilelang.compile(kernel, target=target, target_host=target, execution_backend="tvm_ffi")
    actual = torch.empty((4,), dtype=getattr(torch, dtype))
    executable(actual)
    reference = 2 if dtype == "int32" else 2.75
    torch.testing.assert_close(actual, torch.full_like(actual, reference), rtol=0, atol=0)


def test_vector_input_conversion_keeps_lane_values():
    @C.prim_func
    def kernel(source: C.Tensor((4,), "int8"), output: C.Tensor((4,), "int32")):
        output[C.Ramp(0, 1, 4)] = T.int32x4(source[C.Ramp(0, 1, 4)])

    target = {"kind": "c"}
    executable = tilelang.compile(kernel, target=target, target_host=target, execution_backend="tvm_ffi")
    source = torch.tensor([-128, -1, 0, 127], dtype=torch.int8)
    actual = torch.empty((4,), dtype=torch.int32)
    executable(source, actual)
    torch.testing.assert_close(actual, source.to(torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("constructor", [T.int32, T.int32x4])
def test_dtype_constructor_rejects_a_tensor_value(constructor):
    with pytest.raises(TypeError):

        @C.prim_func
        def kernel(source: C.Tensor((4,), "int32")):
            constructor(source)


def test_dtype_constructor_rejects_a_string():
    with pytest.raises(TypeError):
        T.float32("not a numeric expression")
