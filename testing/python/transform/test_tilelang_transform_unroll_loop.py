import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

import tilelang
import tilelang.testing
from tilelang._compat.stmt_functor import post_order_visit
from tilelang._compat.tensor import TensorView


def _lower(body, parameters):
    function = tirx.Function(parameters, body).with_attr("global_symbol", "main")
    module = tilelang.transform.UnrollLoop()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module["main"]


def _collect(function, kind):
    nodes = []
    post_order_visit(function, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def _unroll(index, extent, body, explicit=False, factor=None):
    annotations = {"pragma_unroll_explicit": explicit}
    if factor is not None:
        annotations["pragma_unroll_factor"] = factor
    return ir.For(index, 0, extent, 1, ir.ForKind.UNROLLED, body, annotations)


def test_unroll_decl_buffer_defs_are_fresh():
    output = TensorView.create("output", [16], "float32")
    alias = TensorView.create("alias", [16], "float32")
    index = ir.Var("index", "int32")
    body = [alias.declare(output.data), alias.store(0, prim.Cast("float32", index))]
    result = _lower(_unroll(index, 2, body, explicit=True), [output.tensor])
    declarations = _collect(result, ir.Bind)
    stores = _collect(result, ir.TensorStore)
    assert len(declarations) == len(stores) == 2
    assert not declarations[0].var.same_as(declarations[1].var)
    assert declarations[0].value.op.name == declarations[1].value.op.name == "tirx.decl_tensor"
    ir.assert_structural_equal(declarations[0].value.args[0], declarations[1].value.args[0])
    assert stores[0].dest.same_as(declarations[0].var)
    assert stores[1].dest.same_as(declarations[1].var)


def test_unroll_explicit_loops_only():
    output = TensorView.create("output", [4], "int32")
    first = ir.Var("first", "int32")
    second = ir.Var("second", "int32")
    result = _lower(
        [_unroll(first, 2, output.store(first, first), explicit=True), _unroll(second, 2, output.store(second + 2, second))],
        [output.tensor],
    )
    assert len(_collect(result, ir.TensorStore)) == 3
    loops = _collect(result, ir.For)
    assert len(loops) == 1
    assert loops[0].kind == ir.ForKind.UNROLLED
    assert loops[0].var.name == "second"


def test_unroll_explicit_loops_preserves_dynamic_and_factor_loops():
    output = TensorView.create("output", [16], "int32")
    extent = ir.Var("extent", "int32")
    first = ir.Var("first", "int32")
    second = ir.Var("second", "int32")
    third = ir.Var("third", "int32")
    result = _lower(
        [
            _unroll(first, 2, output.store(first, first), explicit=True),
            _unroll(second, extent, output.store(second, second)),
            _unroll(third, 8, output.store(third + 8, third), factor=4),
        ],
        [output.tensor, extent],
    )
    loops = _collect(result, ir.For)
    assert [loop.var.name for loop in loops] == ["second", "third"]
    assert all(loop.kind == ir.ForKind.UNROLLED for loop in loops)
    assert int(loops[1].annotations["pragma_unroll_factor"]) == 4


def test_explicit_unroll_rejects_break_targeting_loop():
    output = TensorView.create("output", [2], "int32")
    index = ir.Var("index", "int32")
    body = [output.store(index, index), ir.If(output.load(index) != 0, ir.Break(), None)]
    with pytest.raises(ValueError, match="cannot be fully expanded"):
        _lower(_unroll(index, 2, body, explicit=True), [output.tensor])


def test_non_explicit_unroll_allows_break():
    output = TensorView.create("output", [2], "int32")
    index = ir.Var("index", "int32")
    body = [output.store(index, index), ir.If(output.load(index) != 0, ir.Break(), None)]
    result = _lower(_unroll(index, 2, body), [output.tensor])
    assert len(_collect(result, ir.For)) == 1
    assert _collect(result, ir.For)[0].kind == ir.ForKind.UNROLLED
    assert len(_collect(result, ir.Break)) == 1


def test_explicit_unroll_allows_break_targeting_nested_loop():
    output = TensorView.create("output", [4], "int32")
    outer = ir.Var("outer", "int32")
    inner = ir.Var("inner", "int32")
    body = [output.store(outer * 2 + inner, inner), ir.If(output.load(outer * 2 + inner) != 0, ir.Break(), None)]
    inner_loop = ir.For(inner, 0, 2, 1, ir.ForKind.DEFAULT, body)
    result = _lower(_unroll(outer, 2, inner_loop, explicit=True), [output.tensor])
    loops = _collect(result, ir.For)
    assert len(loops) == 2
    assert all(loop.kind == ir.ForKind.DEFAULT for loop in loops)
    assert all(len(_collect(loop, ir.Break)) == 1 for loop in loops)


if __name__ == "__main__":
    tilelang.testing.main()
