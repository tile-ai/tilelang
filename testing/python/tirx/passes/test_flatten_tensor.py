from __future__ import annotations

import pytest
import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T

from tilelang.transform import _tirx_passes
from tilelang.transform._tirx_passes import FlattenTensor
from tilelang._compat import ir as tlir

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def _transform():
    return tvm_tirx.ir.transform.Sequential(
        [
            _tirx_passes.FlattenTensor(),
            _tirx_passes.StmtSimplify(),
        ]
    )


@pytest.mark.parametrize(
    "strides,origin,begins,extents,expected_begin,expected_extent",
    [
        ([8, 1], 0, [1, 2], [1, 4], 10, 4),
        ([20, 2], 3, [1, 2], [2, 3], 27, 25),
        ([8, -1], 7, [1, 2], [2, 3], 11, 11),
        ([8, 1], 0, [1, 2], [0, 4], 10, 0),
    ],
)
@pytest.mark.parametrize("lower_layouts", [False, True])
def test_flatten_pointer_region_metadata(strides, origin, begins, extents, expected_begin, expected_extent, lower_layouts):
    from tvm_tirx import ir, tirx

    tensor = ir.Var(
        "tensor",
        tirx.TensorType([4, 8], "float32", layout=T.TileLayout(T.S[(4, 8) : strides] + origin @ T.Axis.m)),
    )
    region = ir.TensorRegion(
        tensor,
        [ir.Range.from_min_extent(begin, extent) for begin, extent in zip(begins, extents)],
        ir.TensorRegionType(),
    )
    pointer = ir.Call("tirx.address_of", [ir.TensorLoad(tensor, begins)], attrs={"access_region": region})
    function = tirx.Function([tensor], ir.Evaluate(pointer))
    module = ir.IRModule({"main": tlir.renew_def(function)})
    if lower_layouts:
        with tvm_tirx.target.Target("c"):
            module = _tirx_passes.LowerTensorLayouts()(module)
    result = FlattenTensor()(module)["main"]
    flattened_pointer = result.body[-1].value
    flattened_region = flattened_pointer.attrs["access_region"]
    assert flattened_pointer.args[0].source.same_as(flattened_region.source)
    assert len(flattened_region.region) == 1
    assert int(flattened_region.region[0].begin) == expected_begin
    assert int(flattened_region.region[0].extent) == expected_extent
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.parametrize("lower_layouts", [False, True])
def test_flatten_swizzled_pointer_region_covers_physical_addresses(lower_layouts):
    from tvm_tirx import ir, sym, tirx

    layout = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(16, 64)]))
    tensor = ir.Var("tensor", tirx.TensorType([16, 64], "float32", layout=layout))
    region = ir.TensorRegion(
        tensor,
        [ir.Range.from_min_extent(3, 2), ir.Range.from_min_extent(12, 7)],
        ir.TensorRegionType(),
    )
    pointer = ir.Call("tirx.address_of", [ir.TensorLoad(tensor, [3, 12])], attrs={"access_region": region})
    function = tirx.Function([tensor], ir.Evaluate(pointer))
    module = ir.IRModule({"main": tlir.renew_def(function)})
    if lower_layouts:
        with tvm_tirx.target.Target("c"):
            module = _tirx_passes.LowerTensorLayouts()(module)
    result = FlattenTensor()(module)["main"]
    flattened_pointer = result.body[-1].value
    flattened_region = flattened_pointer.attrs["access_region"]
    assert flattened_pointer.args[0].source.same_as(flattened_region.source)
    assert len(flattened_region.region) == 1
    bounds = flattened_region.region[0]
    analyzer = sym.Analyzer()
    addresses = [int(analyzer.simplify(layout.apply(row, column, shape=[16, 64])["m"])) for row in range(3, 5) for column in range(12, 19)]
    assert 0 <= int(bounds.begin) <= min(addresses)
    assert max(addresses) < int(bounds.begin + bounds.extent) <= 1024
    assert tirx.analysis.check_well_formed(result)
