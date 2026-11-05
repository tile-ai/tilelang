"""Packing preserves Select lanes and complete memory transaction ordering."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

import tilelang
from tilelang.testing.packed import compile_packed_function


def _nodes(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="post")
    return result


def _pack(function):
    before = ir.save_json(function)
    result = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    return result


@pytest.mark.parametrize("vector_condition", [True, False])
def test_select_packing_broadcasts_uniform_arms(vector_condition):
    output = ir.Var("output", tirx.TensorType([4], "int32", data_alignment=16))
    lane = ir.Var("lane", "int32")
    flag = ir.Var("flag", "int32")
    condition = lane == 0 if vector_condition else flag > 0
    on_true = ir.const(10) if vector_condition else lane + 10
    select = ir.prim.Select(condition, on_true, ir.const(-1))
    source = tirx.Function(
        [output, flag], ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, [ir.TensorStore(output, [lane], select)])
    ).with_attr("global_symbol", "main")
    result = _pack(source)
    assert not _nodes(result, ir.For)
    selected = _nodes(result, ir.prim.Select)
    assert len(selected) == 1
    assert selected[0].ty == selected[0].true_value.ty == selected[0].false_value.ty == ir.PrimType("int32x4")
    assert selected[0].condition.ty == ir.PrimType("boolx4" if vector_condition else "bool")
    executable = compile_packed_function(result, c_codegen="compute")
    for flag_value in [0, 1]:
        output = runtime.tensor(np.full(4, 91, dtype="int32"))
        executable(output, flag_value)
        expected = [10, -1, -1, -1] if vector_condition else (list(range(10, 14)) if flag_value else [-1] * 4)
        np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("second_offset", [2, 8])
def test_same_view_transaction_ranges_preserve_scalar_order(second_offset):
    source = ir.Var("source", tirx.TensorType([8], "int32", data_alignment=16))
    output = ir.Var("output", tirx.TensorType([12], "int32", data_alignment=16))
    lane = ir.Var("lane", "int32")
    statements = [
        ir.TensorStore(output, [lane], ir.TensorLoad(source, [lane])),
        ir.TensorStore(output, [lane + second_offset], ir.TensorLoad(source, [lane + 4])),
    ]
    function = tirx.Function([source, output], ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, statements))
    function = function.with_attr("tirx.noalias", True).with_attr("global_symbol", "main")
    result = _pack(function)
    assert bool(_nodes(result, ir.For)) == (second_offset == 2)
    executable = compile_packed_function(result, c_codegen="compute")
    values = np.arange(10, 18, dtype="int32")
    expected = np.full(12, 91, dtype="int32")
    for i in range(4):
        expected[i] = values[i]
        expected[i + second_offset] = values[i + 4]
    actual = runtime.tensor(np.full(12, 91, dtype="int32"))
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
