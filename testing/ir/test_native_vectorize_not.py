"""Packing logical negation re-infers lanes before the result is stored."""

import itertools

import numpy as np
import pytest
import tilelang
import tvm_ffi
from tilelang.testing.packed import compile_packed_function

from tvm_tirx import ir, prim, runtime, tirx


def _nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("kind", ["varying", "nested", "uniform"])
def test_not_packing_preserves_types_and_each_boolean_lane(lanes, kind):
    source = ir.Var("source", tirx.TensorType([lanes], "bool"))
    output = ir.Var("output", source.ty)
    flag = ir.Var("flag", "bool")
    lane = ir.Var("lane", "int32")
    loc = ir.SourceLoc(ir.SourceName("negation.py"), 7, 3, 7, 18)
    operand = flag if kind == "uniform" else ir.TensorLoad(source, [lane])
    value = prim.Not(operand, loc=loc)
    if kind == "nested":
        value = prim.Not(value, loc=loc)
    loop = ir.For(lane, 0, lanes, 1, ir.ForKind.VECTORIZED, [ir.TensorStore(output, [lane], value)])
    # The test supplies distinct input/output arrays. State that contract so
    # memory dependence analysis can admit one packed read/write transaction.
    function = tirx.Function([source, output, flag], loop).with_attr({"global_symbol": "main", "tirx.noalias": True})
    original = ir.save_json(function)
    packed = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == original
    assert tirx.analysis.check_well_formed(packed)
    assert not tirx.analysis.undefined_vars(packed)
    assert not _nodes(packed, ir.For)
    negations = _nodes(packed, prim.Not)
    assert len(negations) == (2 if kind == "nested" else 1)
    for negation in negations:
        assert negation.ty == negation.a.ty == ir.PrimType("bool" if kind == "uniform" else f"boolx{lanes}")
        assert negation.loc.same_as(loc)
    (store,) = _nodes(packed, ir.TensorStore)
    assert store.value.ty == ir.PrimType(f"boolx{lanes}")
    assert bool(_nodes(packed, prim.Broadcast)) == (kind == "uniform")
    ir.assert_structural_equal(packed, ir.load_json(ir.save_json(packed)))
    again = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": packed}))["main"]
    ir.assert_structural_equal(again, packed)

    execute = compile_packed_function(packed, c_codegen="compute")
    # Exhaust all lane combinations so broadcasting one lane cannot pass.
    for values in itertools.product([False, True], repeat=lanes):
        array = np.array(values, dtype="bool")
        for flag_value in (False, True):
            actual = runtime.tensor(np.full(lanes, True, dtype="bool"))
            execute(runtime.tensor(array), actual, flag_value)
            if kind == "uniform":
                expected = np.full(lanes, not flag_value, dtype="bool")
            elif kind == "nested":
                expected = array
            else:
                expected = ~array
            np.testing.assert_array_equal(actual.numpy(), expected)
