"""Nested CPU grids may sink a proven-private, dominating alias group."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx

import tilelang.cpu.transform
from tilelang.testing.packed import compile_packed_function


def _walk(value, kind):
    nodes = []
    tvm_ffi.structural_walk(value, (kind, nodes.append), order="pre")
    return nodes


def _tensor(name, size, scope="global"):
    return ir.Var(name, tirx.TensorType((size,), "int32", scope=scope, layout="default"))


def _allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def _grid(index, body):
    return ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, body, annotations={"tl.cpu_grid_dim": 0})


def _run(out, body, target):
    function = tirx.Function([out], body).with_attr({"target": tvm.target.Target(target), "global_symbol": "main", "tirx.noalias": True})
    module = ir.IRModule.from_expr(function)
    snapshot = ir.save_json(module)
    assert tirx.analysis.check_well_formed(module)
    result = tilelang.cpu.transform.MaterializeCPUParallelGrid()(module)
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(module) == snapshot
    assert all("tl.cpu_grid_dim" not in loop.annotations for loop in _walk(result, ir.For))
    return result


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_nested_grid_sinks_alias_group_without_erasing_lexical_boundary(target):
    out, state, alias = _tensor("out", 16), _tensor("state", 1, "local"), _tensor("alias", 1, "local")
    block, inner = ir.Var("block", "int32"), ir.Var("inner", "int32")
    view = ir.Bind(alias, ir.Call("tirx.decl_tensor", [tirx.tensor_data_ptr(state)], ty_args=[alias.ty]))
    body = [ir.TensorStore(alias, [0], inner + 1), ir.TensorStore(out, [block * 8 + inner], state[0])]
    serial = ir.For(block, 0, 2, 1, ir.ForKind.DEFAULT, [view, _grid(inner, body)])
    lexical = ir.RegionStmt("tl.lexical_scope", [], [], {"explicit_alloc_scope": True}, serial)
    result = _run(out, [_allocate(state), lexical], target)
    (parallel,) = [loop for loop in _walk(result, ir.For) if loop.kind == ir.ForKind.PARALLEL]
    definitions = _walk(parallel.body, ir.Bind)
    assert len(definitions) == 2
    assert definitions[0].var.same_as(state) and definitions[1].var.same_as(alias)
    assert len(_walk(result, ir.Bind)) == 2
    (region,) = _walk(result, ir.RegionStmt)
    assert region.op.name == "tl.lexical_scope" and region.attrs["explicit_alloc_scope"]
    assert parallel in _walk(region.body, ir.For)
    if target == "c":
        actual = tvm.runtime.tensor(np.zeros(16, dtype="int32"))
        compile_packed_function(result)(actual)
        np.testing.assert_array_equal(actual.numpy(), np.tile(np.arange(1, 9, dtype="int32"), 2))


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_nested_grid_does_not_sink_ancestor_allocation_across_thread_context(target):
    out, state, inner = _tensor("out", 8), _tensor("state", 1, "local"), ir.Var("inner", "int32")
    grid = _grid(inner, [ir.TensorStore(state, [0], inner + 1), ir.TensorStore(out, [inner], state[0])])
    context = ir.RegionStmt("tirx.parallel_launch", [], [], {}, grid)
    result = _run(out, [_allocate(state), context], target)
    assert not any(loop.kind == ir.ForKind.PARALLEL for loop in _walk(result, ir.For))
    (allocation,) = _walk(result, ir.Bind)
    assert allocation.var.same_as(state)
    (region,) = _walk(result, ir.RegionStmt)
    assert region.op.name == "tirx.parallel_launch"
    assert not _walk(region.body, ir.Bind)
