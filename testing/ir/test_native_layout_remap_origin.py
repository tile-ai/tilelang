"""LowerTileOp retains physical origins after native contiguous-axis merging."""

import itertools

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.layout import Iter

import tilelang
from tilelang.layout import Layout
from test_native_reducer_contracts import allocate, annotation, attribute, integer


def _address(store):
    tensor = store.dest
    if tensor.ty.layout is not None:
        address = tensor.ty.layout.apply(*store.indices, shape=tensor.ty.shape)["m"]
    else:
        address = ir.const(0, "int64")
        for index, extent in zip(store.indices, tensor.ty.shape):
            address = address * extent + index
    return int(tvm_tirx.sym.Analyzer().simplify(address))


@pytest.mark.parametrize("origin", [0, 5, 13])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("carrier", ["annotation", "attribute", "kernel"])
def test_merged_contiguous_shards_preserve_origin_once(origin, transpose, carrier):
    native_layout = tirx.TileLayout.from_iters(
        [Iter(64, 1, "m")],
        [],
        {"m": ir.const(origin, "int64")},
    )
    shared = ir.Var("versioned", tirx.TensorType([2, 4, 8], "int32", scope="shared.dyn", layout=native_layout))
    layout = Layout([2, 4, 8], lambda stage, row, col: [stage, col, row] if transpose else [stage, row, col])
    points = list(itertools.product(range(2), range(4), range(8)))
    stores = [ir.TensorStore(shared, [integer(axis) for axis in point], integer(index)) for index, point in enumerate(points)]
    if carrier == "annotation":
        body = [allocate(shared), annotation({"layout_map": {shared: layout}}), *stores]
    elif carrier == "attribute":
        body = [allocate(shared), attribute(integer(0), "layout_map", {shared: layout}, stores)]
    else:
        body = [allocate(shared), attribute(integer(0), "tl.kernel", {"layout_map": {shared: layout}, "is_cpu": True}, stores)]
    function = tirx.Function([], body, ir.TupleType([]), ir.DictAttrs({"global_symbol": "main", "target": Target("c")}))
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    lowered = tilelang.transform.LowerTileOp()(ir.IRModule.from_expr(function))["main"]
    assert tirx.analysis.check_well_formed(lowered)
    assert ir.save_json(function) == before
    stores = []
    tvm_ffi.structural_walk(lowered, (ir.TensorStore, stores.append))
    assert len(stores) == len(points)
    for store in stores:
        stage, row, col = points[int(store.value)]
        expected = origin + stage * 32 + (col * 4 + row if transpose else row * 8 + col)
        assert _address(store) == expected


def test_conflicting_kernel_and_inline_layouts_are_rejected():
    shared = ir.Var("shared", tirx.TensorType([2, 3], "int32", scope="shared"))
    identity = Layout([2, 3], lambda row, col: [row, col])
    transpose = Layout([2, 3], lambda row, col: [col, row])
    body = [
        allocate(shared),
        attribute(
            integer(0),
            "tl.kernel",
            {"layout_map": {shared: identity}},
            [annotation({"layout_map": {shared: transpose}}), ir.TensorStore(shared, [integer(1), integer(2)], integer(7))],
        ),
    ]
    function = tirx.Function([], body, ir.TupleType([]), ir.DictAttrs({"target": Target("c")}))
    with pytest.raises(ValueError, match="Conflicting layouts"):
        tilelang.transform.LowerTileOp()(ir.IRModule.from_expr(function))


def test_unresolved_distributed_layout_is_not_treated_as_zero_origin():
    layout = tirx.TileLayout.from_iters([Iter(64, 1, "m")], [Iter(2, 1, "t")], {})
    shared = ir.Var("shared", tirx.TensorType([2, 4, 8], "int32", scope="shared.dyn", layout=layout))
    body = [allocate(shared), annotation({"layout_map": {shared: Layout([2, 4, 8], lambda s, i, j: [s, i, j])}})]
    function = tirx.Function([], body, ir.TupleType([]), ir.DictAttrs({"global_symbol": "main", "target": Target("c")}))
    with pytest.raises(ValueError, match="explicit base offset"):
        tilelang.transform.LowerTileOp()(ir.IRModule.from_expr(function))
