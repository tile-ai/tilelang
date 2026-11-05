"""Analyzer facts follow native For's fixed ``var < end`` predicate."""

import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx

import tilelang
from tilelang.testing.packed import compile_packed_function


def _compile_run(function, factory, arguments=()):
    snapshot = ir.save_json(function)
    module = ir.IRModule({"main": function})
    if factory == "tirx":
        module = tirx.transform.StmtSimplify()(module)
    elif factory == "tilelang":
        module = tilelang.transform.Simplify()(module)
    assert tirx.analysis.check_well_formed(module)
    assert ir.save_json(function) == snapshot
    executable = compile_packed_function(module, c_codegen="compute")
    result = runtime.tensor(np.zeros(3, dtype="int32"))
    executable(result, *arguments)
    return result.numpy()


@pytest.mark.parametrize("factory", ["none", "tirx", "tilelang"])
@pytest.mark.parametrize("step", [-1, 0, 1])
@pytest.mark.parametrize("dynamic", [False, True])
def test_raw_for_range_facts_do_not_assume_python_range(factory, step, dynamic):
    output = ir.Var("output", tirx.TensorType((3,), "int32"))
    index = ir.Var("index", "int32")
    stride = ir.Var("step", "int32") if dynamic else ir.const(step)
    # Exactly four iterations terminate through Break for zero/negative step.
    # The increasing case reaches index == end on its fourth iteration.
    body = [
        ir.TensorStore(output, [0], ir.TensorLoad(output, [0]) + 1),
        ir.TensorStore(output, [1], ir.TensorLoad(output, [1]) + ir.prim.Cast("int32", index < 3)),
        ir.TensorStore(output, [2], ir.TensorLoad(output, [2]) + index),
        ir.If(ir.TensorLoad(output, [0]) == 4, [ir.Break()], None),
    ]
    function = tirx.Function(
        [output, *([stride] if dynamic else [])],
        [ir.For(index, 3, 7, stride, ir.ForKind.DEFAULT, body)],
    ).with_attr("global_symbol", "main")
    expected = np.array([4, 3 if step < 0 else 0, 12 + 6 * step], dtype="int32")
    np.testing.assert_array_equal(_compile_run(function, factory, (step,) if dynamic else ()), expected)


@pytest.mark.parametrize("factory", ["none", "tirx", "tilelang"])
def test_native_descending_endpoints_are_zero_trip(factory):
    output = ir.Var("output", tirx.TensorType((3,), "int32"))
    index = ir.Var("index", "int32")
    function = tirx.Function(
        [output],
        [ir.For(index, 6, 1, -1, ir.ForKind.DEFAULT, [ir.TensorStore(output, [0], 99), ir.Break()])],
    ).with_attr("global_symbol", "main")
    np.testing.assert_array_equal(_compile_run(function, factory), [0, 0, 0])


@pytest.mark.parametrize("factory", ["none", "tirx", "tilelang"])
def test_small_signed_induction_does_not_assume_positive_wrapped_extent(factory):
    output = ir.Var("output", tirx.TensorType((3,), "int32"))
    index = ir.Var("index", "int8")
    function = tirx.Function(
        [output],
        [
            ir.For(
                index,
                ir.const(-100, "int8"),
                ir.const(100, "int8"),
                ir.const(1, "int8"),
                ir.ForKind.DEFAULT,
                [
                    ir.TensorStore(output, [0], ir.TensorLoad(output, [0]) + 1),
                    ir.TensorStore(output, [1], ir.TensorLoad(output, [1]) + ir.prim.Cast("int32", index < ir.const(0, "int8"))),
                ],
            )
        ],
    ).with_attr("global_symbol", "main")
    np.testing.assert_array_equal(_compile_run(function, factory), [200, 100, 0])
