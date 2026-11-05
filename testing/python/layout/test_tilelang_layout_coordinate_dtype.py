"""Layout mapping retains scalar identity, dynamic width and lazy evaluation."""

import numpy as np
import pytest
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang import Fragment, Layout
from tilelang.errors import InternalError
from tilelang.testing.ir import collect_nodes
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("indices", [[], [0, 1]])
def test_layout_mapping_requires_exact_arity(indices):
    layout = Layout((8,), lambda i: i + 1)
    with pytest.raises(InternalError):
        layout.map_forward_index(indices)


def test_zero_dimensional_layout_accepts_empty_indices():
    layout = Layout((), lambda: 3)
    result = layout.map_forward_index([])
    assert len(result) == 1 and int(result[0]) == 3


@pytest.mark.parametrize("indices", [[], [0], [0, 1, 2]])
def test_fragment_mapping_requires_explicit_replica_axis(indices):
    fragment = Fragment((4,), replicate=2, forward_fn=lambda i, rep: (rep * 4 + i, i))
    with pytest.raises(InternalError):
        fragment.map_forward_index(indices)
    with pytest.raises(InternalError):
        fragment.map_forward_thread(indices)


def test_fragment_replica_axis_is_consumed_without_extra_output_axis():
    fragment = Fragment((4,), replicate=2, forward_fn=lambda i, rep: (rep * 4 + i, i))
    assert [int(x) for x in fragment.map_forward_index([1, 2])] == [2]
    assert [int(x) for x in fragment.map_forward_thread([1, 2])] == [6]


def test_same_dtype_mapping_matches_native_index_map_and_keeps_source():
    layout = Layout((8, 4), lambda i, j: [i * 4 + j, i // 2, j % 2])
    before = ir.save_json(layout)
    variables = layout.get_forward_vars()
    p, q = ir.Var("p", "int32"), ir.Var("q", "int32")
    native = tirx.IndexMap(variables, layout.get_forward_index(), None)
    ir.assert_structural_equal(layout.map_forward_index([p, q]), native.map_indices([p, q]))
    assert ir.save_json(layout) == before


def test_layout_mapping_substitutes_simultaneously_once():
    layout = Layout((8, 8), lambda i, j: i * 8 + j)
    i, j = layout.get_forward_vars()
    before = ir.save_json(layout)
    result = layout.map_forward_index([j, i])[0]
    ir.assert_structural_equal(result, j * 8 + i)
    repeated = layout.map_forward_index([i % 5, j])[0]
    ir.assert_structural_equal(repeated, i % 5 * 8 + j)
    assert ir.save_json(layout) == before


def test_lazy_layout_mapping_uses_wide_branches_and_result():
    layout = Layout((16,), lambda i: ir.prim.if_then_else(i < 3, i + 1, i + 2))
    before = ir.save_json(layout)
    p = ir.Var("p", "int64")
    result = layout.map_forward_index([p])[0]
    assert isinstance(result, ir.Call) and result.op.name == "prim.if_then_else"
    assert result.ty.dtype == result.args[1].ty.dtype == result.args[2].ty.dtype == "int64"
    ir.assert_structural_equal(result, ir.prim.if_then_else(p < 3, p + 1, p + 2))
    assert ir.save_json(layout) == before
    assert not collect_nodes(result, ir.prim.Select)


def test_nested_lazy_mapping_propagates_width_to_arithmetic():
    layout = Layout((8, 8), lambda i, j: ir.prim.if_then_else(i < 3, ir.prim.if_then_else(j < 4, i + 1, j + 2), 6) + 2)
    p, q = ir.Var("p", "int64"), ir.Var("q", "int64")
    result = layout.map_forward_index([p, q])[0]
    assert result.ty.dtype == "int64"
    calls = [node for node in collect_nodes(result, ir.Call) if node.op.name == "prim.if_then_else"]
    assert len(calls) == 2
    assert all(node.ty.dtype == node.args[1].ty.dtype == node.args[2].ty.dtype == "int64" for node in calls)
    assert not collect_nodes(result, ir.prim.Select)


def test_layout_mapping_preserves_explicit_narrowing_cast():
    layout = Layout((16,), lambda i: ir.prim.Cast("int16", i + 1))
    p = ir.Var("p", "int64")
    result = layout.map_forward_index([p])[0]
    assert result.ty.dtype == "int16" and result.value.ty.dtype == "int64"
    ir.assert_structural_equal(result, ir.prim.Cast("int16", p + 1))


def test_fragment_lazy_thread_mapping_widens_replica_and_index():
    fragment = Fragment((4,), replicate=2, forward_fn=lambda i, rep: (ir.prim.if_then_else(rep == 0, i, i + 4), i))
    rep, index = ir.Var("rep", "int64"), ir.Var("index", "int64")
    result = fragment.map_forward_thread([rep, index])[0]
    assert result.ty.dtype == "int64"
    assert result.args[1].ty.dtype == result.args[2].ty.dtype == "int64"
    ir.assert_structural_equal(result, ir.prim.if_then_else(rep == 0, index, index + 4))


def test_lazy_layout_runtime_skips_zero_division():
    layout = Layout((16,), lambda i: ir.prim.if_then_else(i != 0, ir.IntImm("int32", 9) // i, i + 11))
    p = ir.Var("p", "int64")
    expression = layout.map_forward_index([p])[0]
    output = ir.Var("output", tirx.TensorType([1], "int64"))
    function = (
        tirx.Function(
            [p, output],
            [ir.TensorStore(output, [ir.IntImm("int32", 0)], expression)],
            ir.PrimType("void"),
        )
        .with_attr("global_symbol", "main")
        .with_attr("target", tvm_tirx.target.Target("c"))
    )
    compiled = compile_packed_function(function, target="c")
    for value in (0, 1, 3, 15):
        result = tvm_tirx.runtime.tensor(np.array([-999], dtype="int64"))
        compiled(value, result)
        assert int(result.numpy()[0]) == (9 // value if value else 11)
