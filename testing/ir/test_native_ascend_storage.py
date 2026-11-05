"""Native Cube storage preserves alias coordinates, capacity, and DMA geometry."""

import pytest
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.ascend.op  # noqa: F401
from tilelang.ascend import language as T
from testing.ascend._ir import substitute
from tilelang.layout import make_ascend_l0c_layout, make_ascend_major_k_layout, make_ascend_major_mn_layout, make_ascend_sf_layout

from test_native_ascend_ops import address_offset, constant, function, strided_tensor, subregion
from test_native_cuda_copy import calls
from test_native_reducer_contracts import allocate, annotation, integer, loop, objects, run_pass, tensor, tileop


def _normalize(value):
    before = ir.save_json(value)
    result = run_pass("NormalizeAscendFractalStorage", value)
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, run_pass("NormalizeAscendFractalStorage", result))
    return result


def _find_copy(value):
    found = calls(value, "tl.tileop.ascend_copy") + calls(value, "tl.tileop.copy")
    assert len(found) == 1
    return found[0]


def _region_extents(value):
    return [r.extent for r in value.region]


def _allocations(value):
    return [bind.var for bind in objects(value, ir.Bind) if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor"]


def _view(owner, shape, dtype=None):
    view = tensor("view", shape, dtype or owner.ty.dtype, owner.ty.scope)
    pointer = owner.data
    if view.ty.dtype != owner.ty.dtype:
        pointer = ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType(view.ty.dtype, owner.ty.scope))
    return view, ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))


def _evaluate(expr, bindings):
    analyzer = tvm_tirx.sym.Analyzer()
    return int(analyzer.simplify(substitute(expr, bindings)))


@pytest.mark.parametrize(
    "shape,dtype,padded",
    [
        ((15, 32), "bfloat16", (16, 32)),
        ((17, 15), "bfloat16", (32, 16)),
        ((16, 32), "bfloat16", (16, 32)),
        ((24, 64), "float32", (32, 64)),
        ((2, 15, 15), "bfloat16", (2, 16, 16)),
    ],
)
def test_canonical_allocation_rounds_only_fractal_axes(shape, dtype, padded):
    owner = tensor("owner", shape, dtype, "shared.l1")
    value = function(
        [],
        [
            allocate(owner),
            annotation({"layout_map": {owner: make_ascend_major_k_layout(owner)}}),
            ir.Evaluate(ir.TensorLoad(owner, [0] * len(shape))),
        ],
    )
    (result,) = _allocations(_normalize(value))
    assert tuple(int(x) for x in result.ty.shape) == padded


@pytest.mark.parametrize("kind", ["view", "reshape", "reinterpret"])
def test_alias_accesses_use_one_canonical_allocation(kind):
    owner = tensor("owner", (15, 15) if kind == "view" else (225,), "uint16" if kind == "reinterpret" else "bfloat16", "shared.l1")
    view, declaration = _view(owner, (15, 15), "bfloat16")
    value = function(
        [],
        [
            allocate(owner),
            declaration,
            annotation({"layout_map": {view: make_ascend_major_k_layout(view)}}),
            ir.TensorStore(view, [1, 0], ir.const(1, "bfloat16")),
        ],
    )
    result = _normalize(value)
    (storage,) = _allocations(result)
    (store,) = objects(result, ir.TensorStore)
    assert store.dest.same_as(storage)
    assert str(storage.ty.dtype) == "bfloat16"
    assert tuple(int(x) for x in storage.ty.shape) == (16, 16)
    assert [int(x) for x in store.indices] == [1, 0]
    assert not calls(result, "tirx.decl_tensor")


@pytest.mark.parametrize("raw", [False, True])
def test_pointer_base_is_reexpressed_in_canonical_indices(raw):
    owner = tensor("owner", (15, 15), "bfloat16", "shared.l1")
    view, declaration = _view(owner, (3, 75))
    row = ir.Var("row", "int32")
    if raw:
        access = ir.Call("tl.ptr_access", [owner.data, integer(15), integer(1), integer(1)])
        body = [ir.Evaluate(access)]
    else:
        access = ir.Call("tl.access_ptr", [ir.TensorLoad(view, [row, 0]), integer(1), integer(1)])
        body = [loop(row, 3, [ir.Evaluate(access)], parallel=False)]
    value = function([], [allocate(owner), declaration, annotation({"layout_map": {owner: make_ascend_major_k_layout(owner)}}), *body])
    result = _normalize(value)
    (access,) = calls(result, "tl.access_ptr")
    load, extent, _ = access.args
    assert load.source.same_as(_allocations(result)[0])
    assert int(load.indices[1]) == 0 and int(extent) == 1
    for i in range(3):
        assert _evaluate(load.indices[0], {row: i}) == (1 if raw else i * 5)


@pytest.mark.parametrize("kind,diagnostic", [("width", "preserve element width"), ("raw-dtype", "canonical operand dtype")])
def test_incompatible_alias_storage_is_rejected(kind, diagnostic):
    owner = tensor("owner", (450 if kind == "width" else 225,), "uint8" if kind == "width" else "uint16", "shared.l1")
    view, declaration = _view(owner, (15, 15), "bfloat16")
    body = [ir.Evaluate(ir.TensorLoad(view, [0, 0]))]
    if kind == "raw-dtype":
        body.append(ir.Evaluate(ir.Call("tl.ptr_access", [owner.data, integer(15), integer(1), integer(1)])))
    value = function([], [allocate(owner), declaration, annotation({"layout_map": {view: make_ascend_major_k_layout(view)}}), *body])
    before = ir.save_json(value)
    with pytest.raises(ValueError, match=diagnostic):
        run_pass("NormalizeAscendFractalStorage", value)
    assert ir.save_json(value) == before


@pytest.mark.parametrize("rows,expected", [(17, 17), (31, 32)])
def test_oob_copy_expands_only_a_complete_destination_axis(rows, expected):
    source = tensor("gm", [31, 128], "bfloat16", "global")
    destination = tensor("l1", [31, 64], "bfloat16", "shared.l1")
    value = function(
        [source],
        [
            allocate(destination),
            annotation({"layout_map": {destination: make_ascend_major_k_layout(destination)}}),
            tileop("ascend_copy", subregion(source, [0, 112], [rows, 64]), subregion(destination, [0, 0], [rows, 64])),
        ],
    )
    copy = _find_copy(_normalize(value))
    assert [int(x) for x in _region_extents(copy.args[0])] == [rows, 64]
    assert [int(x) for x in _region_extents(copy.args[1])] == [expected, 64]


def alias_region_case(shape, starts, extents):
    owner = ir.Var("owner", tirx.TensorType((15, 15), "bfloat16", scope="shared.l1", align=1024))
    view, declaration = _view(owner, shape)
    layout = make_ascend_major_k_layout(owner)
    value = function(
        [],
        [
            allocate(owner),
            declaration,
            annotation({"layout_map": {owner: layout, view: layout.reshape(shape)}, "test_region": subregion(view, starts, extents)}),
            ir.TensorStore(view, starts, ir.const(7, "bfloat16")),
            ir.Evaluate(ir.TensorLoad(view, starts)),
            ir.Evaluate(subregion(view, starts, extents)),
        ],
    )
    return value


def test_alias_indices_metadata_and_regions_share_padded_storage():
    result = _normalize(alias_region_case((225,), [15], [30]))
    (owner,) = _allocations(result)
    assert int(owner.ty.data_alignment) == 1024 and [int(x) for x in owner.ty.shape] == [16, 16]
    for access in [*objects(result, ir.TensorLoad), *objects(result, ir.TensorStore)]:
        tensor_value = access.source if isinstance(access, ir.TensorLoad) else access.dest
        assert tensor_value.same_as(owner) and [int(x) for x in access.indices] == [1, 0]
    for access in objects(result, ir.TensorRegion):
        assert access.source.same_as(owner)
        assert [(int(r.begin), int(r.extent)) for r in access.region] == [(1, 2), (0, 15)]


@pytest.mark.parametrize("alias", [False, True])
def test_native_element_origin_survives_storage_padding_and_layout_lowering(alias):
    owner = strided_tensor("owner", [15, 15], [15, 1], 13, "bfloat16", "shared.l1")
    view = strided_tensor("view", [3, 75], [75, 1], 13, "bfloat16", "shared.l1") if alias else owner
    body = [allocate(owner)]
    if alias:
        body.append(ir.Bind(view, ir.Call("tirx.decl_tensor", [owner.data], ty_args=[view.ty])))
    body.extend(
        [
            annotation({"layout_map": {owner: make_ascend_major_k_layout(owner)}}),
            ir.Evaluate(ir.Call("tl.access_ptr", [ir.TensorLoad(view, [1, 0]), integer(1), integer(1)])),
        ]
    )
    result = _normalize(function([], body))
    (storage,) = _allocations(result)
    assert [int(x) for x in storage.ty.shape] == [16, 16]
    lowered = run_pass("AscendLowerTileOp", result)
    (address,) = calls(lowered, "tl.access_ptr")
    assert address_offset(address) == 13 + (5 if alias else 1) * 16


def test_different_alias_element_origins_are_not_silently_coalesced():
    owner = strided_tensor("owner", [15, 15], [15, 1], 13, "bfloat16", "shared.l1")
    view = strided_tensor("view", [15, 15], [15, 1], 14, "bfloat16", "shared.l1")
    value = function(
        [],
        [
            allocate(owner),
            ir.Bind(view, ir.Call("tirx.decl_tensor", [owner.data], ty_args=[view.ty])),
            annotation({"layout_map": {owner: make_ascend_major_k_layout(owner)}}),
            ir.Evaluate(ir.TensorLoad(view, [0, 0])),
        ],
    )
    with pytest.raises(ValueError, match="same element origin"):
        run_pass("NormalizeAscendFractalStorage", value)


@pytest.mark.parametrize(
    "extents,diagnostic", [([3, 1], "not a contiguous rectangular matrix"), ([3, 80], "crosses an original reshape axis")]
)
def test_unsafe_alias_region_is_rejected(extents, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        run_pass("NormalizeAscendFractalStorage", alias_region_case((3, 75), [0, 0], extents))


def _make_fixpipe_module(
    *,
    dst_dtype="float32",
    src_cols=16,
    dst_cols=16,
    rows=4,
    cols=4,
    dst_strides=None,
    assumptions=(),
    params=(),
    dual_dst_ctl=0,
    src_extents=None,
    dst_extents=None,
    dst_shape=None,
    dst_mins=None,
    copy_op="tl.tileop.ascend_copy",
):
    src = tensor("l0c", (16, src_cols), "float32", "shared.l0c")
    shape = dst_shape or (16, dst_cols)
    dst = (
        strided_tensor("ub", shape, dst_strides, 0, dst_dtype, "shared.dyn")
        if dst_strides
        else tensor("ub", shape, dst_dtype, "shared.dyn")
    )
    body = [allocate(src), allocate(dst), annotation({"layout_map": {src: make_ascend_l0c_layout(src)}})]
    for condition in assumptions:
        body.append(ir.AssertStmt(condition, "test assumption"))
    body.append(
        tileop(
            copy_op.removeprefix("tl.tileop."),
            subregion(src, [0, 0], src_extents or (rows, cols)),
            subregion(dst, dst_mins or [0] * len(shape), dst_extents or (rows, cols)),
            metadata={"dual_dst_ctl": dual_dst_ctl},
        )
    )
    return function(list(params), body)


@pytest.mark.parametrize(
    "dst_dtype,aligned_cols",
    [
        ("float32", 8),
        ("bfloat16", 16),
        ("int8", 32),
    ],
)
@pytest.mark.parametrize("copy_op", ["tl.tileop.copy", "tl.tileop.ascend_copy"])
def test_l0c_to_ub_regions_are_padded_to_32_bytes(dst_dtype, aligned_cols, copy_op):
    mod = _make_fixpipe_module(
        dst_dtype=dst_dtype,
        copy_op=copy_op,
        src_cols=max(16, aligned_cols),
        dst_cols=max(16, aligned_cols),
    )
    normalized = _normalize(mod)
    copy = _find_copy(normalized)

    assert [int(x) for x in _region_extents(copy.args[0])] == [4, aligned_cols]
    assert [int(x) for x in _region_extents(copy.args[1])] == [4, aligned_cols]
    ir.assert_structural_equal(normalized, _normalize(normalized))


def test_l0c_to_ub_dynamic_tail_is_rounded_up():
    actual_n = ir.Var("actual_n", "int32")
    mod = _make_fixpipe_module(
        cols=actual_n,
        assumptions=(actual_n >= 0, actual_n <= 4),
        params=(actual_n,),
    )
    copy = _find_copy(_normalize(mod))
    src_width = _region_extents(copy.args[0])[-1]
    dst_width = _region_extents(copy.args[1])[-1]

    ir.assert_structural_equal(src_width, dst_width)
    for value in range(5):
        assert _evaluate(dst_width, {actual_n: value}) == (8 if value else 0)


@pytest.mark.parametrize("rows", [1, 4])
def test_l0c_to_ub_warns_for_unaligned_ub_row_stride(capfd, rows):
    mod = _make_fixpipe_module(rows=rows, dst_cols=16, dst_strides=(12, 1))
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err

    assert "destination ub row stride is 48 bytes" in stderr
    assert "not a multiple of 32 bytes" in stderr
    assert [int(x) for x in _region_extents(copy.args[1])] == [rows, 8]


@pytest.mark.parametrize(
    "src_extents,dst_extents",
    [((0, 4), (4, 4)), ((4, 0), (4, 4)), ((4, 4), (0, 4)), ((4, 4), (4, 0))],
)
def test_l0c_to_ub_empty_copy_does_not_warn_or_pad(capfd, src_extents, dst_extents):
    mod = _make_fixpipe_module(src_extents=src_extents, dst_extents=dst_extents, dst_strides=(12, 1))
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    assert "Ascend L0C->UB copy:" not in capfd.readouterr().err
    for region, expected in zip(copy.args[:2], (src_extents, dst_extents)):
        assert [int(x) for x in _region_extents(region)] == list(expected)


@pytest.mark.parametrize("known_empty", [False, True])
def test_l0c_to_ub_symbolic_empty_copy_diagnostic(capfd, known_empty):
    n = ir.Var("n", "int32")
    mod = _make_fixpipe_module(
        cols=n,
        dst_strides=(12, 1),
        params=(n,),
        assumptions=(n == 0,) if known_empty else (n >= 0, n <= 4),
    )
    capfd.readouterr()
    _normalize(mod)
    assert ("not a multiple of 32 bytes" in capfd.readouterr().err) == (not known_empty)


@pytest.mark.parametrize("dst_dtype,expected_width", [("bfloat16", 16), ("int8", 4)])
def test_l0c_to_ub_source_capacity_includes_fractal_padding(capfd, dst_dtype, expected_width):
    mod = _make_fixpipe_module(src_cols=4, dst_cols=32, dst_dtype=dst_dtype)
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    assert int(copy.args[0].source.ty.shape[-1]) == 16
    assert int(_region_extents(copy.args[1])[-1]) == expected_width
    assert ("cannot prove padding" in capfd.readouterr().err) == (dst_dtype == "int8")


@pytest.mark.parametrize("dst_cols,stride", [(16, 4), (4, 8)])
def test_l0c_to_ub_padding_respects_shape_and_stride(capfd, dst_cols, stride):
    mod = _make_fixpipe_module(dst_cols=dst_cols, dst_strides=(stride, 1))
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err

    assert "cannot prove padding n_size=4 to 8 elements fits" in stderr
    assert [int(x) for x in _region_extents(copy.args[0])] == [4, 4]
    assert [int(x) for x in _region_extents(copy.args[1])] == [4, 4]


def test_l0c_to_ub_n_split_dual_copy_keeps_its_region():
    mod = _make_fixpipe_module(src_cols=32, src_extents=(4, 32), dst_extents=(4, 16), dual_dst_ctl=2)
    copy = _find_copy(_normalize(mod))

    assert [int(x) for x in _region_extents(copy.args[0])] == [4, 32]
    assert [int(x) for x in _region_extents(copy.args[1])] == [4, 16]


@pytest.mark.parametrize(
    "dual,src_n,dst_n,expected_src,expected_dst",
    [(0, 16, 4, 16, 8), (0, 10, 4, 10, 8), (0, 16, 8, 16, 8), (1, 10, 8, 16, 16), (1, 4, 16, 8, 16)],
)
def test_l0c_to_ub_pads_the_hardware_transfer_width(dual, src_n, dst_n, expected_src, expected_dst):
    mod = _make_fixpipe_module(src_extents=(4, src_n), dst_extents=(2 if dual else 4, dst_n), dual_dst_ctl=dual)
    normalized = _normalize(mod)
    copy = _find_copy(normalized)
    assert [int(x) for x in _region_extents(copy.args[0])] == [4, expected_src]
    assert [int(x) for x in _region_extents(copy.args[1])] == [2 if dual else 4, expected_dst]
    ir.assert_structural_equal(normalized, _normalize(normalized))


@pytest.mark.parametrize("known_unit_stride", [False, True])
def test_l0c_to_ub_symbolic_inner_stride(capfd, known_unit_stride):
    stride = ir.Var("inner_stride", "int32")
    mod = _make_fixpipe_module(dst_strides=(8, stride), params=(stride,), assumptions=(stride == 1,) if known_unit_stride else ())
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err
    assert int(_region_extents(copy.args[1])[-1]) == (8 if known_unit_stride else 4)
    assert ("cannot prove a contiguous 2D destination layout" in stderr) == (not known_unit_stride)


@pytest.mark.parametrize("known_aligned_stride", [False, True])
def test_l0c_to_ub_symbolic_row_stride(capfd, known_aligned_stride):
    stride = ir.Var("row_stride", "int32")
    mod = _make_fixpipe_module(
        dst_strides=(stride, 1),
        params=(stride,),
        assumptions=(stride >= 8, stride % 8 == 0) if known_aligned_stride else (),
    )
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err
    assert int(_region_extents(copy.args[1])[-1]) == (8 if known_aligned_stride else 4)
    assert ("cannot prove it is a multiple of 32 bytes" in stderr) == (not known_aligned_stride)
    assert "it is not a multiple" not in stderr
    assert "undefined" not in stderr


@pytest.mark.parametrize("col_min,expected_width", [(8, 8), (12, 4)])
def test_l0c_to_ub_padding_respects_destination_offset(col_min, expected_width):
    copy = _find_copy(_normalize(_make_fixpipe_module(dst_mins=(0, col_min))))
    assert int(_region_extents(copy.args[1])[-1]) == expected_width
    assert int(copy.args[1].region[-1].begin) == col_min


def test_l0c_to_ub_trailing_unit_axis_uses_the_transfer_axis():
    mod = _make_fixpipe_module(dst_shape=(16, 16, 1), dst_extents=(4, 4, 1))
    copy = _find_copy(_normalize(mod))
    assert [int(x) for x in _region_extents(copy.args[0])] == [4, 8]
    assert [int(x) for x in _region_extents(copy.args[1])] == [4, 8, 1]


def test_l0c_to_ub_unknown_capacity_keeps_the_original_region(capfd):
    n = ir.Var("n", "int32")
    mod = _make_fixpipe_module(cols=n, params=(n,))
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err
    ir.assert_structural_equal(_region_extents(copy.args[1])[-1], n)
    assert "cannot prove padding" in stderr
    assert "undefined" not in stderr


def test_l0c_to_ub_padding_preserves_an_empty_source():
    n = ir.Var("n", "int32")
    mod = _make_fixpipe_module(src_extents=(4, n), params=(n,), assumptions=(n >= 0, n <= 4))
    copy = _find_copy(_normalize(mod))
    width = _region_extents(copy.args[0])[-1]
    for value in (0, 4):
        assert _evaluate(width, {n: value}) == (8 if value else 0)


@pytest.mark.parametrize(
    "dtype,major_mn,transpose,source,destination,expected_source,expected_destination,pitch",
    [
        ("bfloat16", False, False, [24, 16], [24, 16], [32, 16], [32, 16], 2),
        ("bfloat16", False, False, [24, 16], [64, 32], [32, 16], [32, 16], 4),
        ("bfloat16", False, True, [24, 16], [16, 24], [32, 16], [16, 32], 1),
        ("float32", False, False, [17, 1], [17, 1], [32, 8], [32, 8], 2),
        ("float8_e4m3fn", False, False, [16, 17], [16, 17], [16, 32], [16, 32], 1),
        ("float8_e4m3fn", True, False, [17, 17], [17, 17], [32, 32], [32, 32], 2),
    ],
)
def test_l0_copy_regions_match_instruction_transfer(
    dtype, major_mn, transpose, source, destination, expected_source, expected_destination, pitch
):
    src = tensor("src", [64, 64], dtype, "shared.l1")
    dst = tensor("dst", [64, 64], dtype, "shared.l0a")
    layout = make_ascend_major_mn_layout if major_mn else make_ascend_major_k_layout
    value = function(
        [],
        [
            allocate(src),
            allocate(dst),
            annotation({"layout_map": {src: layout(src), dst: layout(dst)}}),
            tileop(
                "ascend_copy", subregion(src, [0, 0], source), subregion(dst, [0, 0], destination), metadata={"transpose": int(transpose)}
            ),
        ],
    )
    result = _normalize(value)
    operation = _find_copy(result)
    assert [constant(x) for x in _region_extents(operation.args[0])] == expected_source
    assert [constant(x) for x in _region_extents(operation.args[1])] == expected_destination
    (load,) = calls(run_pass("AscendLowerTileOp", result), "tl.ascend_load_cbuf_to_ca")
    assert constant(load.args[7]) == pitch


@pytest.mark.parametrize("empty_operand", ["source", "destination"])
def test_l0_copy_normalization_preserves_dynamic_empty_transfer(empty_operand):
    n = ir.Var("n", "int32")
    rows = ir.prim.min(ir.prim.max(n, -1), 24)
    src = tensor("src", [32, 16], "bfloat16", "shared.l1")
    dst = tensor("dst", [32, 16], "bfloat16", "shared.l0a")
    src_rows, dst_rows = (rows, 32) if empty_operand == "source" else (1, ir.prim.min(ir.prim.max(n, -1), 16))
    value = function(
        [n],
        [
            allocate(src),
            allocate(dst),
            annotation({"layout_map": {src: make_ascend_major_k_layout(src), dst: make_ascend_major_k_layout(dst)}}),
            tileop("ascend_copy", subregion(src, [0, 0], [src_rows, 16]), subregion(dst, [0, 0], [dst_rows, 16])),
        ],
    )
    result = _normalize(value)
    operation = _find_copy(result)
    for count in (-1, 0, 1, 24):
        expected = 0 if count <= 0 else (32 if count == 24 and empty_operand == "source" else 16)
        assert [_evaluate(operand.region[0].extent, {n: count}) for operand in operation.args[:2]] == [expected, expected]
    lowered = run_pass("AscendLowerTileOp", result)
    assert calls(lowered, "tl.ascend_load_cbuf_to_ca")
    (guard,) = objects(lowered, ir.If)
    for count in (-1, 0, 1, 24):
        assert bool(_evaluate(guard.condition, {n: count})) == (count > 0)


@pytest.mark.parametrize("dtype,source,expected", [("uint8", [24, 3], [32, 4]), ("uint16", [24, 3], [32, 3])])
def test_sf_source_uses_scale_pair_transfer_geometry(dtype, source, expected):
    src = tensor("src", [32, 4], dtype, "shared.l1")
    sf = tensor("sf", [32, 4], dtype, "shared.l0a.sf")
    value = function(
        [],
        [
            allocate(src),
            allocate(sf),
            annotation({"layout_map": {src: make_ascend_sf_layout(src)}}),
            tileop("ascend_copy", subregion(src, [0, 0], source), subregion(sf, [0, 0], source)),
        ],
    )
    operation = _find_copy(_normalize(value))
    assert [constant(x) for x in _region_extents(operation.args[0])] == expected
    assert [constant(x) for x in _region_extents(operation.args[1])] == source


@pytest.mark.parametrize("dual,src_n,dst_n,expected_n", [(False, 4, 4, 8), (False, 10, 4, 8), (True, 10, 8, 16)])
def test_real_gemm_dsl_fixpipe_descriptor_is_aligned(dual, src_n, dst_n, expected_n):
    dst_m = 2 if dual else 4

    @T.prim_func
    def main(A: T.Tensor((16, 16), "bfloat16"), B: T.Tensor((16, 16), "bfloat16"), C: T.Tensor((dst_m, dst_n), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((16, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((dst_m, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, accum, transpose_B=True, clear_accum=True)
            if dual:
                T.dual_copy(accum[:4, :src_n], ub[:, :dst_n])
            else:
                T.copy(accum[:4, :src_n], ub[:, :dst_n])
            T.copy(ub[:, :dst_n], C)

    result = main.with_attr("target", tvm_tirx.target.Target("ascend"))
    # Source block layout annotations are owned by the materialized kernel,
    # as in the Ascend pipeline used by actual user kernels.
    result = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        launch_dim_tags=["cthread"],
    )(ir.IRModule({"main": result}))["main"]
    for name in ("AscendLayoutInference", "NormalizeAscendFractalStorage", "AscendInsertOOBPadding", "AscendLowerTileOp"):
        result = run_pass(name, result)
        assert tirx.analysis.check_well_formed(result)
    descriptors = calls(result, "tl.ascend_copy_matrix_cc_to_ub")
    assert [tuple(constant(copy.args[i]) for i in (3, 5, 7)) for copy in descriptors] == [(expected_n, 16, int(dual))]
