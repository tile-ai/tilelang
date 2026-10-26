import ctypes

import numpy as np
import pytest
import tvm_ffi

from tvm_tirx import ir, runtime, tirx


@pytest.mark.parametrize("dtype,value", [("int32", -19), ("float32", 1.25), ("bool", True)])
def test_packed_call_returns_scalar_value(dtype, value):
    name = f"test.c_codegen.packed_return_{dtype}"
    tvm_ffi.register_global_func(name, lambda: value, override=True)
    output = ir.Var("output", tirx.TensorType([1], dtype))
    result = ir.Call("tirx.call_packed", [ir.StringImm(name)], ty=dtype)
    function = tirx.Function([output], ir.TensorStore(output, [ir.const(0)], result)).with_attr("global_symbol", "packed_scalar")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), dtype)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [value])


@pytest.mark.parametrize("pointer", [0, 4096])
def test_packed_call_returns_pointer_value(pointer):
    name = "test.c_codegen.packed_pointer"
    tvm_ffi.register_global_func(name, lambda: ctypes.c_void_p(pointer), override=True)
    output = ir.Var("output", tirx.TensorType([1], "uint64"))
    result = ir.Call("tirx.call_packed", [ir.StringImm(name)], ty=ir.PtrType("void"))
    value = tirx.reinterpret(result, ty="uint64")
    function = tirx.Function([output], ir.TensorStore(output, [ir.const(0)], value)).with_attr("global_symbol", "packed_pointer")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "uint64")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [pointer])


def test_nested_packed_arguments_use_their_stack_offsets():
    tvm_ffi.register_global_func("test.c_codegen.add", lambda first, second: first + second, override=True)
    output = ir.Var("output", tirx.TensorType([1], "int32"))

    def add(first, second):
        return ir.Call("tirx.call_packed", [ir.StringImm("test.c_codegen.add"), first, second], ty="int32")

    value = add(ir.const(3), add(ir.const(5), ir.const(7)))
    function = tirx.Function([output], ir.TensorStore(output, [ir.const(0)], value)).with_attr("global_symbol", "packed_nested")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [15])
