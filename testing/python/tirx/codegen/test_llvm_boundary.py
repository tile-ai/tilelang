import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.codegen import Executable
from tvm_tirx.target import Target

from tilelang.transform import _tirx_passes


def build(function):
    if not runtime.enabled("llvm"):
        pytest.skip("LLVM is not enabled")
    target = Target("llvm", host="llvm")
    function = function.with_attr("global_symbol", "main")
    module = _tirx_passes.BindTarget(target)(ir.IRModule({"main": function}))
    for lowering in (
        _tirx_passes.LowerTensorLayouts,
        _tirx_passes.FlattenTensor,
        _tirx_passes.MakePackedAPI,
        _tirx_passes.LowerTVMBuiltin,
        _tirx_passes.LowerIntrin,
    ):
        module = lowering()(module)
    assert tirx.analysis.check_well_formed(module)
    return Executable(tvm_ffi.get_global_func("target.build.tilelang_llvm")(module, target))


@pytest.mark.parametrize("step", [-3, -1, 0, 1, 3])
def test_dynamic_step_and_direction(step):
    output = ir.Var("output", tirx.TensorType([10], "int32"))
    begin = ir.Var("begin", "int32")
    end = ir.Var("end", "int32")
    stride = ir.Var("stride", "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, begin, end, stride, ir.ForKind.DEFAULT, ir.TensorStore(output, [index], index))
    executable = build(tirx.Function([output, begin, end, stride], loop))
    first, stop = (9, -1) if step < 0 else (0, 10)
    result = runtime.tensor(np.full(10, -1, dtype="int32"))
    executable(result, first, stop, step)
    expected = np.full(10, -1, dtype="int32")
    for index in range(first, stop, step) if step else []:
        expected[index] = index
    np.testing.assert_array_equal(result.numpy(), expected)


def test_descending_continue_and_break():
    output = ir.Var("output", tirx.TensorType([10], "int32"))
    index = ir.Var("index", "int32")
    body = [
        ir.If(index == 7, ir.Continue(), None),
        ir.If(index == 3, ir.Break(), None),
        ir.TensorStore(output, [index], index),
    ]
    executable = build(tirx.Function([output], ir.For(index, 9, -1, -2, ir.ForKind.DEFAULT, body)))
    result = runtime.tensor(np.full(10, -1, dtype="int32"))
    executable(result)
    expected = np.full(10, -1, dtype="int32")
    expected[[9, 5]] = [9, 5]
    np.testing.assert_array_equal(result.numpy(), expected)


def test_effectful_bounds_are_evaluated_once():
    output = ir.Var("output", tirx.TensorType([10], "int32"))
    index = ir.Var("index", "int32")
    evaluations = []
    name = "tl.testing.llvm_loop_control"

    def control(value):
        evaluations.append(value)
        return value

    def call(value):
        return ir.Call("tirx.call_packed", [ir.StringImm(name), ir.const(value)], ty="int32")

    tvm_ffi.register_global_func(name, control)
    try:
        loop = ir.For(index, call(9), call(-1), call(-3), ir.ForKind.DEFAULT, ir.TensorStore(output, [index], index))
        executable = build(tirx.Function([output], loop))
        result = runtime.tensor(np.full(10, -1, dtype="int32"))
        executable(result)
        assert evaluations == [9, -1, -3]
        expected = np.full(10, -1, dtype="int32")
        expected[[9, 6, 3, 0]] = [9, 6, 3, 0]
        np.testing.assert_array_equal(result.numpy(), expected)
    finally:
        tvm_ffi.registry.remove_global_func(name)
