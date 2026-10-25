import itertools
import sys

import pytest
import tvm_ffi
from tvm_tirx import ir, sym

from tilelang._compat.tensor import TensorView


def coordinates(shape):
    return itertools.product(*(range(int(extent)) for extent in shape))


def evaluate(layout, indices):
    return tuple(int(value) for value in layout(*indices))


@pytest.mark.parametrize("kind", ["layout", "fragment"])
def test_layout_normalizes_constant_extents(native_layouts, kind):
    extent = ir.prim.Max(ir.const(0), ir.const(32))
    if kind == "layout":
        layout = native_layouts.Layout([extent], lambda index: index)
    else:
        layout = native_layouts.Fragment([extent], lambda index, replica: (index * 4 + replica, 0), replicate=4)
        assert isinstance(layout.replicate_size, ir.IntImm)
        assert int(layout.replicate_size) == 4
    assert all(isinstance(size, ir.IntImm) and int(size) == 32 for size in layout.get_input_shape())


@pytest.mark.parametrize("kind", ["layout", "fragment", "partial", "major_k", "major_mn", "l0c", "sf"])
def test_native_layout_script_preserves_type_and_mapping(native_layouts, kind, capfd):
    import tilelang
    from tvm_tirx.script import tirx as script
    from tvm_tirx.script.printer import PrinterConfig
    from tvm_tirx.script.printer.scriptable import _script

    if kind == "layout":
        layout = native_layouts.Layout([4, 8], lambda row, column: row * 8 + column)
    elif kind in ("fragment", "partial"):
        layout = native_layouts.Fragment([4, 8], lambda row, column, replica: (replica * 32 + row * 8 + column, 0), replicate=2)
        if kind == "partial":
            layout = native_layouts.PartialFragment.from_fragment(layout)
    else:
        tensor = TensorView.create("tensor", [32, 64], "float16", scope="shared.l1").tensor
        layout = getattr(native_layouts, "make_ascend_" + kind + "_layout")(tensor)
    capfd.readouterr()
    source = _script(layout, PrinterConfig())
    assert "printer has no Doc hook" not in capfd.readouterr().err
    restored = eval(source, {"tilelang": tilelang, "T": script, "I": ir})
    assert type(restored) is type(layout)
    ir.assert_structural_equal(restored, layout)


def test_native_layout_script_keeps_external_symbolic_identity(native_layouts):
    import tilelang
    from tvm_tirx import tirx
    from tvm_tirx.script import from_source
    from tvm_tirx.script import tirx as script

    extent = ir.Var("extent", "int32")
    layout = native_layouts.Layout([4, extent], lambda row, column: row * extent + column)
    function = tirx.Function([extent], ir.Evaluate(0), ret_type=ir.TupleType([])).with_attr("test_layout", layout)
    source = function.script()
    restored = from_source(source, extra_vars={"tilelang": tilelang, "T": script})
    ir.assert_structural_equal(restored, function)


@pytest.mark.parametrize("shape", [(8, 8), (3, 4, 5), (16,), (1, 4)])
def test_layout_native_indices_and_inverse(native_layouts, shape):
    def linearize(*indices):
        result = 0
        for index, extent in zip(indices, shape):
            result = result * extent + index
        return result

    layout = native_layouts.Layout(shape, linearize)
    inverse = layout.inverse()
    assert all(isinstance(variable, ir.Var) for variable in layout.get_forward_vars())
    for ordinal, indices in enumerate(coordinates(shape)):
        assert evaluate(layout, indices) == (ordinal,)
        assert evaluate(inverse, (ordinal,)) == indices
    assert "tvm" not in sys.modules


def test_layout_repeat_expand_and_reshape(native_layouts):
    base = native_layouts.Layout([4, 8], lambda row, column: row * 8 + column)
    expanded = base.expand([3, 2])
    repeated = base.repeat(0, 3)
    reshaped = base.reshape([8, 4])
    for batch, group, row, column in coordinates([3, 2, 4, 8]):
        assert evaluate(expanded, (batch, group, row, column)) == (batch, group, row * 8 + column)
    for row, column in coordinates([12, 8]):
        assert evaluate(repeated, (row, column)) == (row // 4, row % 4 * 8 + column)
    for row, column in coordinates([8, 4]):
        assert evaluate(reshaped, (row, column)) == (row * 4 + column,)
    assert base.expand([]) is base


@pytest.mark.parametrize("shape,threads", [((2, 6, 32), 128), ((8, 16, 1, 2), 128), ((3, 10, 8), 16)])
def test_inverse_handles_thread_splits_across_reshape_axes(native_layouts, shape, threads):
    def flattened(*indices):
        value = 0
        for index, extent in zip(indices, shape):
            value = value * extent + index
        return [value // threads, value % threads]

    layout = native_layouts.Layout(shape, flattened)
    inverse = layout.inverse()
    for logical in coordinates(shape):
        assert evaluate(inverse, evaluate(layout, logical)) == logical


def test_fragment_index_eliminates_replica_before_normalizing_thread_map(native_layouts):
    fragment = native_layouts.Fragment(
        [8, 32], forward_thread_fn=lambda row, column, replica: (row * 64 + replica * 4 + column // 8) % 128, replicate=16
    )
    assert tuple(int(extent) for extent in fragment.get_output_shape()) == (32,)
    for row, column in coordinates([8, 32]):
        assert evaluate(fragment, (0, row, column)) == (row // 2 * 8 + column % 8,)


def test_layout_reinterpret_preserves_storage_bits(native_layouts):
    base = native_layouts.Layout([4, 8], lambda row, column: row * 8 + column)
    narrower = base.reshape([4, 16], rescale_num=32, rescale_den=16)
    wider = narrower.reshape([4, 8], rescale_num=16, rescale_den=32)
    for row, column in coordinates([4, 16]):
        assert evaluate(narrower, (row, column)) == (row * 16 + column,)
    assert wider.is_equal(base)


def test_layout_region_bounds_and_symbolic_shape(native_layouts):
    extent = ir.Var("extent", "int32")
    layout = native_layouts.Layout([extent, 8], lambda row, column: [row, column])
    region = layout.map_region_bounds([ir.Range(2, 6), ir.Range(1, 4)])
    assert [(int(axis.begin), int(axis.extent)) for axis in region] == [(2, 4), (1, 3)]
    expanded = layout.expand([3])
    assert expanded.get_input_shape()[1].same_as(extent)


@pytest.mark.parametrize("origin", [0, 8])
def test_xor_swizzle_output_shape_includes_all_physical_coordinates(native_layouts, origin):
    layout = native_layouts.Layout(
        [64, 32],
        lambda row, column: [0, row, ((column // 8) ^ ((row % 8) // 2)) * 8 + column % 8 + origin],
    )
    assert [int(extent) for extent in layout.get_output_shape()] == [1, 64, 32 + origin]
    shape = layout.get_output_shape()
    for coordinate in coordinates((64, 32)):
        assert all(0 <= index < int(extent) for index, extent in zip(evaluate(layout, coordinate), shape, strict=True))


@pytest.mark.parametrize("replicas", [1, 2, 4])
def test_fragment_native_thread_ownership(native_layouts, replicas):
    def thread(row, column, replica=0):
        return row * 4 + column // 2 + replica * 32

    fragment = native_layouts.Fragment([8, 8], forward_thread_fn=thread, replicate=replicas)
    assert int(fragment.get_thread_size()) == 32 * replicas
    for replica, row, column in coordinates([replicas, 8, 8]):
        indices = [row, column] if replicas == 1 else [replica, row, column]
        assert evaluate(fragment, indices) == (column % 2,)
        assert int(fragment.map_forward_thread(indices)[0]) == row * 4 + column // 2 + replica * 32


def test_fragment_forward_function_and_partial_semantics(native_layouts):
    fragment = native_layouts.Fragment([8], forward_fn=lambda index, replica: (index + replica * 8, [0]), replicate=4)
    partial = native_layouts.PartialFragment.from_fragment(fragment)
    assert isinstance(partial, native_layouts.PartialFragment)
    assert not partial.is_equal(fragment)
    assert not fragment.is_equal(partial)
    assert partial.as_post_collective().is_equal(fragment)
    assert int(partial.combine_size) == 4
    combined = native_layouts.PartialFragment([8], forward_fn=lambda index, replica: (index + replica * 8, (0,)), replicate=4, combine=2)
    assert int(combined.combine_size) == 2
    steps = tvm_ffi.get_global_func("tl.PartialFragment_combine_steps")(combined)
    assert [tuple(int(value) for value in step) for step in steps] == [(16, 8)]


@pytest.mark.parametrize(
    "constructor",
    ["make_quarter_bank_swizzled_layout", "make_half_bank_swizzled_layout", "make_full_bank_swizzled_layout"],
)
def test_swizzle_native_tensor_view_boundary(native_layouts, constructor):
    make_layout = getattr(native_layouts, constructor)
    tensor = TensorView.create("tile", [8, 64], "float16", scope="shared")
    layout = make_layout(tensor)
    assert make_layout(tensor.tensor).is_equal(layout)
    assert make_layout(tensor.load([0, 0])).is_equal(layout)
    assert make_layout(tensor.region([ir.Range(0, 8), ir.Range(0, 64)])).is_equal(layout)
    expanded = TensorView.create("expanded", [3, 2, 8, 64], "float16", scope="shared")
    assert make_layout(expanded).is_equal(layout.expand([3, 2]))
    assert len({evaluate(layout, index) for index in coordinates([8, 64])}) == 8 * 64
    with pytest.raises(TypeError):
        make_layout(ir.Var("scalar", "int32"))


@pytest.mark.parametrize("shape,strides", [((8, 4), (4, 1)), (((2, 4), 8), ((1, 2), 8)), ((4, 8), (1, 8))])
def test_cute_layout_algebra_and_roundtrip(native_layouts, shape, strides):
    cute = native_layouts.cute
    layout = cute.make_layout(shape, strides)
    restored = cute.Layout.parse(str(layout))
    assert restored.shape == shape
    assert restored.stride == strides
    converted = layout.to_tilelang()
    recovered = cute.Layout.from_tilelang(converted)
    assert recovered is not None
    for indices in coordinates(converted.get_input_shape()):
        assert int(recovered(indices)) == evaluate(converted, indices)[0]
    loaded = ir.load_json(ir.save_json(layout))
    assert str(loaded) == str(layout)


def test_cute_dynamic_layout_uses_native_sym(native_layouts):
    cute = native_layouts.cute
    extent = ir.Var("extent", "int32")
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    layout = cute.make_layout((8, extent), (extent, 1))
    result = layout((row, column))
    assert isinstance(result, ir.Expr)
    assert sym.Analyzer().can_prove_equal(result, row * extent + column)


@pytest.mark.parametrize("atom", ["ld_32dp32b", "ld_16dp64b", "ld_16dp128b", "ld_16dp256b"])
def test_tcgen05_native_layout_plan(native_layouts, atom):
    getter = tvm_ffi.get_global_func(f"tl.get_tcgen05_meta_{atom}")
    tile = native_layouts.cute.Layout.parse("(128,64):(1@0,1@1)")
    plan = tvm_ffi.get_global_func("tl.ExpandTcgen05Layout")(getter(), tile, 128)
    assert plan is not None
    owners = set()
    for datapath, column in coordinates([128, 64]):
        owner, slot = native_layouts.cute.to_python(native_layouts.cute.from_python(plan.fragment((datapath, column))) + (0, 0))
        assert 0 <= int(owner) < 128
        owners.add((int(owner), int(slot)))
    assert len(owners) == 128 * 64


@pytest.mark.parametrize("kind", ["Layout", "Fragment", "PartialFragment"])
def test_layout_serialization_preserves_coordinate_binding(native_layouts, kind):
    if kind == "Layout":
        layout = native_layouts.Layout([4, 8], lambda row, column: [column, row])
        domain = [4, 8]
    else:
        layout = getattr(native_layouts, kind)(
            [4, 8], forward_thread_fn=lambda row, column, replica: row * 4 + column // 2 + replica * 16, replicate=2
        )
        domain = [2, 4, 8]
    restored = ir.load_json(ir.save_json(layout))
    assert type(restored) is type(layout)
    ir.assert_structural_equal(restored, layout)
    assert restored.is_equal(layout)
    for indices in coordinates(domain):
        assert evaluate(restored, indices) == evaluate(layout, indices)
        if kind != "Layout":
            assert int(restored.map_forward_thread(indices)[0]) == int(layout.map_forward_thread(indices)[0])


def test_layout_serialization_shares_external_shape_variables(native_layouts):
    extent = ir.Var("extent", "int32")
    layout = native_layouts.Layout([extent, 8], lambda row, column: row * 8 + column)
    loaded_extent, loaded_layout = ir.load_json(ir.save_json(tvm_ffi.Array([extent, layout])))
    assert loaded_layout.get_input_shape()[0].same_as(loaded_extent)
    assert evaluate(loaded_layout, [3, 5]) == (29,)
