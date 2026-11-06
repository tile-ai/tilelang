"""Mutable assumptions justify bounds only while their observed values survive."""

from tilelang.language import _annotation_codec

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

import tilelang
import tilelang.language as T
from tilelang.testing.ir import collect_calls
from tilelang.testing.packed import compile_packed_function


CPU = tvm_tirx.target.Target("c")


def _tensor(name, shape):
    return ir.Var(name, tirx.TensorType(shape, "int32", layout="default", align=4))


def _assume(index):
    return ir.Evaluate(prim.assume(prim.all(index >= 0, index < 4)))


def _module(kind):
    data, index, output = _tensor("data", (4,)), _tensor("index", (1,)), _tensor("output", (1,))
    params = [data, index, output]
    read = ir.TensorLoad(index, [0])
    fact = _assume(read)
    write = ir.TensorStore(output, [0], ir.TensorLoad(data, [read]))
    body = [fact]
    if kind == "store":
        body.append(ir.TensorStore(index, [0], ir.const(5, "int32")))
    elif kind == "alias":
        alias = _tensor("alias", (1,))
        body.extend(
            [
                ir.Bind(alias, ir.Call("tirx.decl_tensor", [index.data], ty_args=[alias.ty])),
                ir.TensorStore(alias, [0], ir.const(5, "int32")),
            ]
        )
    elif kind == "opaque":
        body.append(ir.Evaluate(tirx.call_extern("mutable_index_external", index.data, ty="void")))
    elif kind == "branch":
        flag = ir.Var("flag", "bool")
        params.append(flag)
        body = [ir.If(flag, [fact], None)]
    elif kind == "loop":
        loop = ir.Var("i", "int32")
        body = [ir.For(loop, 0, 2, 1, ir.ForKind.DEFAULT, [fact, ir.TensorStore(index, [0], loop + 4)])]
    elif kind == "same_name":
        other = _tensor("index", (1,))
        params.append(other)
        write = ir.TensorStore(output, [0], ir.TensorLoad(data, [ir.TensorLoad(other, [0])]))
    elif kind != "plain":
        raise AssertionError(kind)
    body.append(write)
    function = tirx.Function(params, body, ir.PrimType("void")).with_attr({"global_symbol": "main", "target": CPU})
    return ir.IRModule({"main": function})


def _guard(module):
    original = ir.save_json(module)
    result = tilelang.transform.LegalizeSafeMemoryAccess()(module)
    assert ir.save_json(module) == original
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def test_unchanged_memory_assumption_eliminates_its_exact_bounds():
    result = _guard(_module("plain"))
    assert not collect_calls(result, op="prim.if_then_else")
    executable = compile_packed_function(result)
    for value in (0, 1, 3):
        data = np.arange(4, dtype="int32") + 10
        index = np.array([value], dtype="int32")
        output = np.full(1, -99, dtype="int32")
        executable(*(tvm_ffi.from_dlpack(array) for array in (data, index, output)))
        np.testing.assert_array_equal(output, data[value : value + 1])


@pytest.mark.parametrize("kind", ["store", "alias", "opaque", "branch", "loop", "same_name"])
def test_a_memory_fact_cannot_survive_a_possible_change_or_escape_its_scope(kind):
    result = _guard(_module(kind))
    assert collect_calls(result, op="prim.if_then_else")


@pytest.mark.parametrize("kind", ["store", "alias"])
def test_mutation_after_assume_preserves_padding_for_the_new_index(kind):
    executable = compile_packed_function(_guard(_module(kind)))
    data = np.arange(4, dtype="int32") + 10
    index = np.array([2], dtype="int32")
    output = np.full(1, -99, dtype="int32")
    executable(*(tvm_ffi.from_dlpack(array) for array in (data, index, output)))
    np.testing.assert_array_equal(index, [5])
    np.testing.assert_array_equal(output, [0])


def test_volatile_scope_does_not_turn_a_previous_load_into_a_stable_fact():
    pass

    @T.prim_func
    def main(data: T.Tensor((4,), "int32"), index: T.Tensor((1,), "int32"), output: T.Tensor((1,), "int32")):
        with _annotation_codec.attr(index, "volatile_scope", 1):
            T.assume(index[0] >= 0 and index[0] < 4)
            output[0] = data[index[0]]

    result = _guard(ir.IRModule({"main": main}))
    assert collect_calls(result, op="prim.if_then_else")


@pytest.mark.parametrize("different_slot", [False, True])
def test_memory_snapshot_preserves_arithmetic_and_access_identity(different_slot):
    data, index, output = _tensor("data", (4,)), _tensor("index", (2,)), _tensor("output", (1,))
    observed = index[0]
    queried = index[1] if different_slot else observed
    function = tirx.Function(
        [data, index, output],
        [
            ir.Evaluate(prim.assume(prim.all(observed >= 1, observed <= 4))),
            ir.TensorStore(output, [0], data[queried - 1]),
        ],
        ir.PrimType("void"),
    ).with_attr("global_symbol", "main")
    result = _guard(ir.IRModule({"main": function}))
    assert bool(collect_calls(result, op="prim.if_then_else")) == different_slot
    executable = compile_packed_function(result)
    values = np.arange(4, dtype="int32") + 10
    for position in (1, 2, 4):
        indices = np.array([position, 5], dtype="int32")
        actual = np.full(1, -99, dtype="int32")
        executable(*(tvm_ffi.from_dlpack(array) for array in (values, indices, actual)))
        np.testing.assert_array_equal(actual, [0 if different_slot else values[position - 1]])
