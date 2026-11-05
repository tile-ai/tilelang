"""Version layouts preserve logical index types and wide physical pitches."""

import pytest
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import ir, tirx

from testing.ascend._ir import allocated_tensors, kernel
from testing.ascend.auto_schedule._scheduled_ir import unit


def versioned(shape, index_type, dtype="float32", layout=None):
    extents = [ir.IntImm(index_type, value) for value in shape]
    storage = ir.Var("storage", tirx.TensorType(extents, dtype, scope="shared.dyn", layout=layout))
    i = ir.Var("i", "int32")
    store = ir.TensorStore(storage, [0] * len(shape), ir.const(1, dtype))
    loop = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [unit(store)], annotations={"multi_buffer_eligible": [storage]})
    before = kernel(
        unit(loop, core=None),
        buffers=[storage],
        annotations={"tl.buffer_versions_map": {storage: 3}},
        allocation_stage=0,
    )
    assert tirx.analysis.check_well_formed(before)
    serialized = ir.save_json(before)
    after = transform.MaterializeMultiBuffer()(before)
    assert ir.save_json(before) == serialized
    assert tirx.analysis.check_well_formed(after)
    (result,) = allocated_tensors(after, include_scheduled=True)
    return storage, result


@pytest.mark.parametrize("index_type", ["int32", "int64", "uint32"])
def test_compact_versions_keep_exact_extent_types(index_type):
    storage, result = versioned((16, 64), index_type)
    assert tuple(int(value) for value in result.ty.shape) == (3, 16, 64)
    layout = result.ty.layout
    assert isinstance(layout, tirx.TileLayout)
    assert not layout.replica
    assert len(layout.shard) == len(result.ty.shape)
    for dimension, iterator in zip(result.ty.shape, layout.shard):
        ir.assert_structural_equal(dimension, iterator.extent)
        assert str(dimension.ty) == index_type
    for original, extent in zip(storage.ty.shape, result.ty.shape[1:]):
        assert original.same_as(extent)
    analyzer = tvm_tirx.sym.Analyzer()
    assert [int(analyzer.simplify(iterator.stride)) for iterator in layout.shard] == [1024, 64, 1]
    assert all(str(iterator.stride.ty) == "int64" for iterator in layout.shard)


@pytest.mark.parametrize(
    "index_type, shape, pitch",
    [("int32", (65536, 65536), 1 << 32), ("uint32", (1 << 31, 2), 1 << 32), ("int64", (1 << 32, 2), 1 << 33)],
)
def test_physical_stride_does_not_wrap_at_the_logical_index_width(index_type, shape, pitch):
    _, result = versioned(shape, index_type, "uint8")
    layout = result.ty.layout
    analyzer = tvm_tirx.sym.Analyzer()
    assert int(analyzer.simplify(layout.shard[0].stride)) == pitch
    assert str(layout.shard[0].stride.ty) == "int64"
    for dimension, iterator in zip(result.ty.shape, layout.shard):
        ir.assert_structural_equal(dimension, iterator.extent)
        assert str(dimension.ty) == index_type


def test_physical_span_overflow_remains_an_error():
    with pytest.raises(ValueError, match="overflows int64"):
        versioned((1 << 62, 4), "int64", "uint8")
