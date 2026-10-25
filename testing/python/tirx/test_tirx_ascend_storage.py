import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang._compat import stmt_functor

pytestmark = pytest.mark.usefixtures("ascend_codegen")


def _nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _apply(name, body, parameters=()):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    result = tvm_ffi.get_global_func("tl.transform." + name)()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    assert tirx.analysis.check_well_formed(restored)
    return result


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _pointer(tensor, offset=0, extent=16):
    return ir.Call(
        "tirx.ptr_byte_offset",
        [ir.Call("tirx.tensor_data_ptr", [tensor]), ir.const(offset, "int64")],
        ty=ir.PtrType(tensor.ty.dtype, tensor.ty.scope),
        attrs={"tl.access_extent": extent, "tl.access_mask": 3},
    )


@pytest.mark.parametrize("parameter", [False, True])
def test_fp4_storage_retypes_pairs_without_rescaling_byte_offsets(parameter):
    tensor = ir.Var("storage", tirx.TensorType([32], "float4_e2m1fn", scope="global" if parameter else "shared"))
    body = ([] if parameter else [_allocate(tensor)]) + [ir.Evaluate(_pointer(tensor, 7, 16))]
    result = _apply("RewriteFp4ToFp4x2", body, [tensor] if parameter else [])
    rewritten = result.params[0] if parameter else _nodes(result, ir.Bind)[0].var
    assert rewritten.ty.dtype.dtype.bits == 4 and rewritten.ty.dtype.dtype.lanes == 2
    assert int(rewritten.ty.shape[0]) == 16
    pointer = next(call for call in _nodes(result, ir.Call) if call.op.name == "tirx.ptr_byte_offset")
    assert int(pointer.args[1]) == 7
    assert int(pointer.attrs["tl.access_extent"]) == 8
    assert pointer.ty.element_type.dtype.lanes == 2
    assert int(pointer.attrs["tl.access_mask"]) == 3


def test_fp4_byte_backed_view_keeps_its_existing_geometry():
    storage = ir.Var("bytes", tirx.TensorType([16], "int8", scope="shared"))
    view = ir.Var("view", tirx.TensorType([32], "float4_e2m1fn", scope="shared"))
    declaration = ir.Bind(view, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [storage])], ty_args=[view.ty]))
    result = _apply("RewriteFp4ToFp4x2", [_allocate(storage), declaration, ir.Evaluate(_pointer(view, 7, 16))])
    rewritten_view = _nodes(result, ir.Bind)[1].var
    assert rewritten_view.ty.dtype.dtype.lanes == 1
    assert int(rewritten_view.ty.shape[0]) == 32
    pointer = next(call for call in _nodes(result, ir.Call) if call.op.name == "tirx.ptr_byte_offset")
    assert int(pointer.args[1]) == 7
    assert int(pointer.attrs["tl.access_extent"]) == 16


@pytest.mark.parametrize("lanes", [2, 8])
def test_fp4_vector_values_preserve_logical_lane_count(lanes):
    source = ir.Var("source", tirx.TensorType([32], "float4_e2m1fn"))
    destination = ir.Var("destination", source.ty)
    indices = [ir.prim.Ramp(ir.const(4), ir.const(1), lanes)]
    body = ir.TensorStore(destination, indices, ir.TensorLoad(source, indices))
    result = _apply("RewriteFp4ToFp4x2", body, [source, destination])
    load = _nodes(result, ir.TensorLoad)[0]
    store = _nodes(result, ir.TensorStore)[0]
    assert load.ty.dtype.lanes == lanes
    assert store.value.ty.dtype.lanes == lanes
    if lanes == 2:
        assert int(load.indices[0]) == 2
    else:
        assert int(load.indices[0].base) == 2
        assert int(load.indices[0].lanes) == lanes // 2


def test_fp4_address_of_does_not_load_or_duplicate_a_scalar_value():
    tensor = ir.Var("source", tirx.TensorType([32], "float4_e2m1fn"))
    address = ir.Call("tirx.address_of", [ir.TensorLoad(tensor, [4])])
    result = _apply("RewriteFp4ToFp4x2", ir.Evaluate(address), [tensor])
    load = _nodes(result, ir.TensorLoad)[0]
    assert int(load.indices[0]) == 2
    assert load.ty.dtype.lanes == 2


@pytest.mark.parametrize("store", [False, True])
def test_fp4_scalar_value_access_is_not_silently_retyped_to_a_pair(store):
    tensor = ir.Var("source", tirx.TensorType([32], "float4_e2m1fn"))
    load = ir.TensorLoad(tensor, [0])
    body = ir.TensorStore(tensor, [2], load) if store else ir.Evaluate(load)
    with pytest.raises(ValueError, match="one scalar fp4 value"):
        _apply("RewriteFp4ToFp4x2", body, [tensor])


@pytest.mark.parametrize("shape,extent", [(31, 16), (32, 15)])
def test_fp4_rejects_partial_byte_storage_or_access(shape, extent):
    tensor = ir.Var("source", tirx.TensorType([shape], "float4_e2m1fn"))
    with pytest.raises(ValueError, match="must be even"):
        _apply("RewriteFp4ToFp4x2", ir.Evaluate(_pointer(tensor, 0, extent)), [tensor])


def test_fp4_aliases_and_metadata_share_retyped_definitions():
    tensor = ir.Var("storage", tirx.TensorType([32], "float4_e2m1fn", scope="shared"))
    view = ir.Var("view", tensor.ty)
    declaration = ir.Bind(view, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [tensor])], ty_args=[view.ty]))
    region = ir.RegionStmt(
        "tirx.scope",
        [],
        [],
        {"buffers": {tensor: [view]}},
        [_allocate(tensor), declaration, ir.Evaluate(_pointer(view))],
    )
    result = _apply("RewriteFp4ToFp4x2", region)
    rewritten_region = _nodes(result, ir.RegionStmt)[0]
    storage_binding, view_binding = _nodes(result, ir.Bind)
    assert storage_binding.var.ty.dtype.dtype.lanes == 2
    assert view_binding.var.ty.dtype.dtype.lanes == 2
    assert rewritten_region.attrs["buffers"][storage_binding.var][0].same_as(view_binding.var)


def _version_scope(tensor, body, versions=2):
    return ir.RegionStmt("tirx.scope", [], [], {"tl.buffer_version": {tensor: versions}}, body)


def test_versioned_ub_slots_preserve_numerical_accesses_and_shifted_aliases():
    storage = ir.Var("storage", tirx.TensorType([2, 5], "int32", scope="shared"))
    alias = ir.Var("second_version", tirx.TensorType([5], "int32", scope="shared"))
    byte_alias = ir.Var("bytes", tirx.TensorType([40], "uint8", scope="shared"))
    output = ir.Var("output", tirx.TensorType([12], "int32"))
    version = ir.Var("version", "int32")
    index = ir.Var("index", "int32")
    initialize = ir.For(
        version,
        0,
        2,
        1,
        ir.ForKind.DEFAULT,
        ir.For(index, 0, 5, 1, ir.ForKind.DEFAULT, ir.TensorStore(storage, [version, index], version * 100 + index)),
    )
    second = ir.Bind(alias, ir.Call("tirx.decl_tensor", [_pointer(storage, 20, 5)], ty_args=[alias.ty]))
    byte_view = ir.Bind(byte_alias, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [storage])], ty_args=[byte_alias.ty]))
    copy = [ir.TensorStore(output, [position], ir.TensorLoad(storage, [position // 5, position % 5])) for position in range(10)]
    body = _version_scope(
        storage,
        [
            _allocate(storage),
            initialize,
            second,
            byte_view,
            *copy,
            ir.TensorStore(output, [10], ir.TensorLoad(alias, [1])),
            ir.TensorStore(output, [11], ir.prim.Cast("int32", ir.TensorLoad(byte_alias, [24]))),
        ],
    )
    result = _apply("RewriteAscendBufferVersionLayout", body, [output])
    allocation = _nodes(result, ir.Bind)[0]
    assert [int(extent) for extent in allocation.var.ty.shape] == [16]
    assert not any("tl.buffer_version" in region.attrs for region in _nodes(result, ir.RegionStmt))
    assert [int(store.indices[0]) for store in _nodes(result, ir.TensorStore) if store.dest.same_as(output)][-2:] == [10, 11]
    actual = runtime.empty([12], "int32")
    tirx.compile(result, target="c")(actual)
    np.testing.assert_array_equal(actual.numpy(), [0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 101, 101])


def test_versioned_pointer_offsets_are_rewritten_once_through_aliases():
    storage = ir.Var("storage", tirx.TensorType([2, 5], "int32", scope="shared"))
    base = ir.Var("base", ir.PtrType("int32", "shared"))
    bind_pointer = ir.Bind(base, _pointer(storage, 20, 5))
    pointer = ir.Call(
        "tirx.ptr_byte_offset",
        [base, ir.const(4, "int64")],
        ty=base.ty,
        attrs={"tl.access_extent": 2, "tl.access_mask": 1},
    )
    result = _apply("RewriteAscendBufferVersionLayout", _version_scope(storage, [_allocate(storage), bind_pointer, ir.Evaluate(pointer)]))
    pointers = [call for call in _nodes(result, ir.Call) if call.op.name == "tirx.ptr_byte_offset"]
    assert [int(call.args[1]) for call in pointers] == [32, 36]
    assert [int(call.attrs["tl.access_extent"]) for call in pointers] == [5, 2]


def test_versioned_pointer_cannot_cross_padding_between_slots():
    storage = ir.Var("storage", tirx.TensorType([2, 5], "int32", scope="shared"))
    body = _version_scope(storage, [_allocate(storage), ir.Evaluate(_pointer(storage, 16, 2))])
    with pytest.raises(ValueError, match="crosses a 32-byte-aligned version slot"):
        _apply("RewriteAscendBufferVersionLayout", body)


@pytest.mark.parametrize("shape,versions", [([2, 8], 2), ([20], 2), ([5], 1)])
def test_version_layout_preserves_aligned_and_count_only_manual_storage(shape, versions):
    storage = ir.Var("storage", tirx.TensorType(shape, "int32", scope="shared"))
    result = _apply("RewriteAscendBufferVersionLayout", _version_scope(storage, [_allocate(storage)], versions))
    assert [int(extent) for extent in _nodes(result, ir.Bind)[0].var.ty.shape] == shape


def test_version_layout_does_not_require_a_ub_allocation_for_l1_parameters():
    storage = ir.Var("storage", tirx.TensorType([2, 5], "int32", scope="shared.l1"))
    result = _apply("RewriteAscendBufferVersionLayout", _version_scope(storage, ir.Evaluate(0)), [storage])
    assert result.params[0].same_as(storage)


def _fractal_scope(body, layouts):
    return ir.RegionStmt("tirx.scope", [], [], {"layout_map": layouts}, body)


def _copy(source, destination, source_shape, destination_shape, attrs=None):
    regions = [
        ir.TensorRegion(tensor, [ir.Range(0, extent) for extent in shape], ir.TensorRegionType())
        for tensor, shape in [(source, source_shape), (destination, destination_shape)]
    ]
    return ir.Evaluate(ir.Call("tl.tileop.ascend_copy", regions, attrs=attrs or {}, ty=ir.TupleType([])))


@pytest.mark.parametrize(
    "shape,dtype,padded",
    [
        ([15, 32], "bfloat16", [16, 32]),
        ([17, 15], "bfloat16", [32, 16]),
        ([16, 32], "bfloat16", [16, 32]),
        ([24, 64], "float32", [32, 64]),
        ([2, 15, 15], "bfloat16", [2, 16, 16]),
    ],
)
def test_fractal_allocation_rounds_only_matrix_axes(native_layouts, shape, dtype, padded):
    storage = ir.Var("storage", tirx.TensorType(shape, dtype, scope="shared.l1"))
    body = _fractal_scope(
        [_allocate(storage), ir.Evaluate(ir.TensorLoad(storage, [0] * len(shape)))],
        {storage: native_layouts.make_ascend_nz_layout(storage)},
    )
    result = _apply("NormalizeAscendFractalStorage", body)
    allocation = _nodes(result, ir.Bind)[0].var
    assert [int(extent) for extent in allocation.ty.shape] == padded
    assert _nodes(result, ir.TensorLoad)[0].source.same_as(allocation)
    second = _apply("NormalizeAscendFractalStorage", result.body)
    ir.assert_structural_equal(result, second)


@pytest.mark.parametrize("owner_shape,owner_dtype", [([15, 15], "bfloat16"), ([225], "bfloat16"), ([225], "uint16")])
def test_fractal_aliases_share_one_native_definition(native_layouts, owner_shape, owner_dtype):
    storage = ir.Var("storage", tirx.TensorType(owner_shape, owner_dtype, scope="shared.l1", data_alignment=1024))
    view = ir.Var("view", tirx.TensorType([15, 15], "bfloat16", scope="shared.l1"))
    declaration = ir.Bind(view, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [storage])], ty_args=[view.ty]))
    body = _fractal_scope(
        [_allocate(storage), declaration, ir.TensorStore(view, [1, 0], ir.const(7, "bfloat16"))],
        {view: native_layouts.make_ascend_nz_layout(view)},
    )
    result = _apply("NormalizeAscendFractalStorage", body)
    assert len(_nodes(result, ir.Bind)) == 1
    allocation = _nodes(result, ir.Bind)[0].var
    assert [int(extent) for extent in allocation.ty.shape] == [16, 16]
    assert allocation.ty.dtype == ir.PrimType("bfloat16")
    assert allocation.ty.data_alignment == 1024
    store = _nodes(result, ir.TensorStore)[0]
    assert store.dest.same_as(allocation)
    assert [int(index) for index in store.indices] == [1, 0]


def test_fractal_raw_pointer_uses_logical_indices_without_double_byte_scaling(native_layouts):
    storage = ir.Var("storage", tirx.TensorType([15, 15], "bfloat16", scope="shared.l1"))
    body = _fractal_scope(
        [_allocate(storage), ir.Evaluate(_pointer(storage, 30, 1))],
        {storage: native_layouts.make_ascend_nz_layout(storage)},
    )
    result = _apply("NormalizeAscendFractalStorage", body)
    pointer = next(call for call in _nodes(result, ir.Call) if call.op.name == "tl.access_ptr")
    assert [int(index) for index in pointer.args[0].indices] == [1, 0]
    assert int(pointer.args[1]) == 1


@pytest.mark.parametrize(
    "shape,mins,extents,message", [([3, 75], [0, 0], [3, 1], "contiguous rectangular"), ([3, 75], [0, 0], [3, 80], "original reshape axis")]
)
def test_fractal_alias_region_rejects_holes_and_overlap(native_layouts, shape, mins, extents, message):
    storage = ir.Var("storage", tirx.TensorType([15, 15], "bfloat16", scope="shared.l1"))
    alias = ir.Var("alias", tirx.TensorType(shape, "bfloat16", scope="shared.l1"))
    declaration = ir.Bind(alias, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [storage])], ty_args=[alias.ty]))
    region = ir.TensorRegion(alias, [ir.Range(start, start + extent) for start, extent in zip(mins, extents)], ir.TensorRegionType())
    body = _fractal_scope([_allocate(storage), declaration, ir.Evaluate(region)], {storage: native_layouts.make_ascend_nz_layout(storage)})
    with pytest.raises(RuntimeError, match=message):
        _apply("NormalizeAscendFractalStorage", body)


@pytest.mark.parametrize("dtype,width", [("float32", 8), ("bfloat16", 16), ("int8", 32)])
def test_fixpipe_padding_uses_destination_byte_width(native_layouts, dtype, width):
    source = ir.Var("source", tirx.TensorType([16, 32], "float32", scope="shared.l0c"))
    destination = ir.Var("destination", tirx.TensorType([16, 32], dtype, scope="shared"))
    body = _fractal_scope(
        [_allocate(source), _allocate(destination), _copy(source, destination, [4, 4], [4, 4])],
        {source: native_layouts.make_ascend_l0c_layout(source)},
    )
    result = _apply("NormalizeAscendFractalStorage", body)
    copy = next(call for call in _nodes(result, ir.Call) if call.op.name == "tl.tileop.ascend_copy")
    assert [[int(axis.extent) for axis in region.region] for region in copy.args] == [[4, width], [4, width]]
    ir.assert_structural_equal(result, _apply("NormalizeAscendFractalStorage", result.body))


@pytest.mark.parametrize(
    "dtype,major_mn,transpose,source_shape,destination_shape,expected_source,expected_destination",
    [
        ("bfloat16", False, False, [24, 16], [24, 16], [32, 16], [32, 16]),
        ("bfloat16", False, False, [24, 16], [64, 32], [32, 16], [32, 16]),
        ("bfloat16", False, True, [24, 16], [16, 24], [32, 16], [16, 32]),
        ("float32", False, False, [17, 1], [17, 1], [32, 8], [32, 8]),
        ("float8_e4m3fn", False, False, [16, 17], [16, 17], [16, 32], [16, 32]),
        ("float8_e4m3fn", True, False, [17, 17], [17, 17], [32, 32], [32, 32]),
    ],
)
def test_l0_copy_normalization_preserves_transfer_geometry(
    native_layouts,
    dtype,
    major_mn,
    transpose,
    source_shape,
    destination_shape,
    expected_source,
    expected_destination,
):
    source = ir.Var("source", tirx.TensorType([64, 64], dtype, scope="shared.l1"))
    destination = ir.Var("destination", tirx.TensorType([64, 64], dtype, scope="shared.l0a"))
    layout = native_layouts.make_ascend_major_mn_layout if major_mn else native_layouts.make_ascend_nz_layout
    body = _fractal_scope(
        [
            _allocate(source),
            _allocate(destination),
            _copy(source, destination, source_shape, destination_shape, {"transpose": int(transpose)}),
        ],
        {source: layout(source), destination: layout(destination)},
    )
    result = _apply("NormalizeAscendFractalStorage", body)
    copy = next(call for call in _nodes(result, ir.Call) if call.op.name == "tl.tileop.ascend_copy")
    assert [[int(axis.extent) for axis in region.region] for region in copy.args] == [expected_source, expected_destination]
    ir.assert_structural_equal(result, _apply("NormalizeAscendFractalStorage", result.body))
