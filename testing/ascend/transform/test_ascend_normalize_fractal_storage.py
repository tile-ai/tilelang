"""Canonical Cube allocations, aliases and logical-to-padded indices."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import (
    access_tensor,
    allocated_tensors,
    annotation,
    annotations,
    sblock,
    sblock_metadata,
    sblocks,
    single_statement,
    substitute,
    tensor,
    walk_ir,
)

import pytest
import tilelang
import tvm_tirx as tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0c_layout, make_ascend_nz_layout, make_ascend_major_mn_layout, make_ascend_sf_layout
from tvm_tirx import tirx
from testing.ascend._ir import calls, nodes, kernel, copy
from tilelang.language._native import buffer as tensor_spec


def _strided_tensor(shape, dtype, *, name="buffer", scope="global", strides=None, elem_offset=0, align=0):
    return ir.Var(name, tensor_spec(shape, dtype, strides=strides, elem_offset=elem_offset, scope=scope, align=align).as_type())


def _view_binding(view, owner):
    return ir.Bind(view, ir.Call("tirx.decl_tensor", [tirx.tensor_data_ptr(owner)], ty_args=[view.ty]))


def _module(owner, view, body):
    statements = [] if owner.same_as(view) else [_view_binding(view, owner)]
    statements += [annotation({"layout_map": {view: make_ascend_nz_layout(view)}}), body]
    block = sblock("root", statements, buffers=[owner])
    return tvm.IRModule({"main": tirx.Function(tirx.analysis.undefined_vars(block), block)})


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
    buffer = tensor(shape, dtype, name="tile", scope="shared.l1")
    before = _module(buffer, buffer, ir.Evaluate(buffer[tuple(0 for _ in shape)]))
    after = transform.NormalizeAscendFractalStorage()(before)
    allocation = allocated_tensors(single_statement(after["main"].body))[0]
    assert tuple(int(x) for x in allocation.ty.shape) == padded
    assert allocation.name == buffer.name
    assert tuple(int(x) for x in buffer.ty.shape) == shape
    assert nodes(after, ir.TensorLoad)[0].source.same_as(allocation)
    tvm.ir.assert_structural_equal(after, transform.NormalizeAscendFractalStorage()(after))


@pytest.mark.parametrize("kind", ["view", "reshape", "reinterpret"])
def test_alias_accesses_use_one_canonical_allocation(kind):
    owner_shape = (15, 15) if kind == "view" else (225,)
    owner = tensor(owner_shape, "uint16" if kind == "reinterpret" else "bfloat16", name="owner", scope="shared.l1")
    view = tensor((15, 15), "bfloat16", name="view", scope="shared.l1")
    before = _module(owner, view, ir.TensorStore(view, [1, 0], ir.const(1, "bfloat16")))
    after = transform.NormalizeAscendFractalStorage()(before)
    allocation = allocated_tensors(single_statement(after["main"].body))[0]
    (store,) = nodes(after, ir.TensorStore)
    assert access_tensor(store).same_as(allocation)
    assert allocation.name == owner.name and len(allocated_tensors(after)) == 1
    assert str(allocation.ty.dtype.dtype) == "bfloat16"
    assert tuple(int(x) for x in allocation.ty.shape) == (16, 16)
    assert tuple(int(x) for x in store.indices) == (1, 0)


@pytest.mark.parametrize("raw", [False, True], ids=["reshaped-access", "raw-pointer"])
def test_pointer_base_is_reexpressed_in_canonical_indices(raw):
    owner = tensor((15, 15), "bfloat16", name="owner", scope="shared.l1")
    alias = tensor((3, 75), "bfloat16", name="alias", scope="shared.l1")
    row = ir.Var("row", "int32")
    pointer = ir.Call("tl.ptr_access", [tirx.tensor_data_ptr(owner), 15, 1, 1]) if raw else T.access_ptr(alias[row, 0], "r", extent=1)
    body = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("consume"), pointer], ty="int32"))
    before = _module(
        owner,
        owner,
        body
        if raw
        else ir.SeqStmt(
            [_view_binding(alias, owner), ir.For(row, ir.const(0), ir.const(0) + ir.const(3), ir.const(1), ir.ForKind.DEFAULT, body)]
        ),
    )
    after = transform.NormalizeAscendFractalStorage()(before)
    (pointer,) = calls(after, "tl.access_ptr")
    load, extent, _ = pointer.args
    assert access_tensor(load).same_as(allocated_tensors(single_statement(after["main"].body))[0])
    analyzer = tvm.sym.Analyzer()
    assert analyzer.can_prove_equal(load.indices[0], 1 if raw else row * 5)
    assert int(load.indices[1]) == 0 and int(extent) == 1


@pytest.mark.parametrize(
    "kind,message",
    [
        ("width", "Cube aliases must preserve element width"),
        ("raw-dtype", "Cube raw pointer must use the canonical operand dtype"),
    ],
)
def test_incompatible_alias_storage_is_rejected(kind, message):
    owner = tensor((450 if kind == "width" else 225,), "uint8" if kind == "width" else "uint16", scope="shared.l1")
    view = tensor((15, 15), "bfloat16", name="view", scope="shared.l1")
    body = ir.Evaluate(view[0, 0])
    if kind == "raw-dtype":
        body = ir.SeqStmt(
            [
                body,
                ir.Evaluate(
                    ir.Call(
                        "tirx.call_extern",
                        [ir.StringImm("consume"), ir.Call("tl.ptr_access", [tirx.tensor_data_ptr(owner), 15, 1, 1])],
                        ty="int32",
                    )
                ),
            ]
        )
    before = _module(owner, view, body)
    with pytest.raises(ValueError, match=message):
        transform.NormalizeAscendFractalStorage()(before)


@pytest.mark.parametrize("rows,expected_rows", [(17, 17), (31, 32)])
def test_oob_copy_expands_only_a_complete_destination_axis(rows, expected_rows):
    src = tensor((31, 128), "bfloat16")
    dst = tensor((31, 64), "bfloat16", scope="shared.l1")
    read = ir.TensorRegion(src, [tvm.ir.Range(0, rows), tvm.ir.Range(112, 176)], ir.TensorRegionType())
    write = ir.TensorRegion(dst, [tvm.ir.Range(0, rows), tvm.ir.Range(0, 64)], ir.TensorRegionType())
    before = _module(dst, dst, ir.Evaluate(T.copy(read, write)))
    after = transform.NormalizeAscendFractalStorage()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    assert [int(axis.extent) for axis in copy.args[0].region] == [rows, 64]
    assert [int(axis.extent) for axis in copy.args[1].region] == [expected_rows, 64]


def _make_alias_region_module(view_shape, mins, extents):
    owner = tensor((15, 15), "bfloat16", name="owner", scope="shared.l1")
    view = tensor(view_shape, "bfloat16", name="view", scope="shared.l1")
    layout = tilelang.layout.make_ascend_nz_layout(owner)
    ranges = [tvm.ir.Range.from_min_extent(start, extent) for start, extent in zip(mins, extents)]
    region = ir.TensorRegion(view, ranges, ir.TensorRegionType())
    load = ir.TensorLoad(view, mins)
    region_call = ir.TensorRegion(view, ranges, ir.TensorRegionType())
    body = ir.SeqStmt([ir.TensorStore(view, mins, ir.const(7, "bfloat16")), ir.Evaluate(load), ir.Evaluate(region_call)])
    # Definitions dominate the native region's explicit read/write operands.
    block = sblock(
        "root",
        [
            _view_binding(view, owner),
            annotation({"layout_map": {owner: layout, view: layout.reshape(view_shape)}}),
            sblock("accesses", body, reads=[region], writes=[region]),
        ],
        buffers=[owner],
    )
    return tvm.IRModule({"main": tirx.Function([], block)})


def test_alias_indices_and_access_regions_use_the_same_padded_storage():
    normalized = tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((225,), (15,), (30,)))
    root = single_statement(normalized["main"].body)
    owner = allocated_tensors(root)[0]
    metadata = sblock_metadata(sblocks(root, "accesses")[0])
    alias = access_tensor(metadata["reads"][0])
    assert owner.same_as(alias)
    assert tuple(int(x) for x in owner.ty.shape) == tuple(int(x) for x in alias.ty.shape) == (16, 16)
    assert [(int(r.begin), int(r.extent)) for r in metadata["reads"][0].region] == [(1, 2), (0, 15)]
    tvm.ir.assert_structural_equal(metadata["reads"][0], metadata["writes"][0])
    for node in [*nodes(root, ir.TensorStore), *nodes(root, ir.TensorLoad)]:
        assert tuple(int(x) for x in node.indices) == (1, 0)
        assert tuple(int(x) for x in access_tensor(node).ty.shape) == (16, 16)
    tvm.ir.assert_structural_equal(normalized, tilelang.ascend.transform.NormalizeAscendFractalStorage()(normalized))


def test_alias_normalization_preserves_allocation_properties():
    owner = _strided_tensor((225,), "uint16", name="owner", scope="shared.l1", align=1024, elem_offset=0)
    view = tensor((15, 15), "bfloat16", name="view", scope="shared.l1")
    layout = tilelang.layout.make_ascend_nz_layout(view)
    block = sblock(
        "root",
        [
            _view_binding(view, owner),
            annotation(
                {
                    "tl.tensor_spec": {owner: {"offset_factor": 8, "buffer_type": "", "axis_separators": []}},
                    "layout_map": {view: layout, owner: layout.reshape((225,))},
                }
            ),
            ir.Evaluate(ir.TensorLoad(view, [1, 0])),
        ],
        buffers=[owner],
    )
    mod = tvm.IRModule({"main": tirx.Function([], block)})
    normalized = tilelang.ascend.transform.NormalizeAscendFractalStorage()(mod)
    allocation = allocated_tensors(single_statement(normalized["main"].body))[0]
    assert allocation.name == owner.name
    assert allocation.data_alignment == 1024
    (spec,) = [fields["tl.tensor_spec"][allocation] for fields in annotations(normalized) if "tl.tensor_spec" in fields]
    assert int(spec["offset_factor"]) == 8
    assert str(allocation.ty.dtype.dtype) == "bfloat16"
    assert tuple(int(x) for x in allocation.ty.shape) == (16, 16)
    assert allocation.same_as(nodes(normalized, ir.TensorLoad)[0].source)


def test_non_rectangular_alias_region_is_rejected():
    # Three logical elements in different rows cannot become an 11-row DMA.
    with pytest.raises(ValueError, match="not a contiguous rectangular matrix"):
        tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((3, 75), (0, 0), (3, 1)))


def test_overlapping_alias_region_is_rejected():
    # The 3x80 source box repeats logical elements across the 75-element row
    # stride, even though its bounding box is also 240 elements (16x15).
    with pytest.raises(ValueError, match="crosses an original reshape axis"):
        tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((3, 75), (0, 0), (3, 80)))


def _make_region(buffer, extents, access_mask, mins=None):
    return ir.TensorRegion(
        buffer,
        [ir.Range.from_min_extent(start, extent) for start, extent in zip(mins if mins is not None else [0] * len(extents), extents)],
        ir.TensorRegionType(),
    )


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
    src = tensor((16, src_cols), "float32", name="l0c", scope="shared.l0c")
    dst = _strided_tensor(dst_shape if dst_shape is not None else (16, dst_cols), dst_dtype, name="ub", scope="shared", strides=dst_strides)
    annotations = tvm_ffi.convert({"dual_dst_ctl": ir.IntImm("int32", dual_dst_ctl)})
    copy = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"),
        tvm.ir.Op.get(copy_op),
        [
            _make_region(src, src_extents if src_extents is not None else (rows, cols), 1),
            _make_region(dst, dst_extents if dst_extents is not None else (rows, cols), 2, dst_mins),
        ],
        annotations,
        ir.UnknownLoc(),
    )
    body = ir.Evaluate(copy)
    # Native analyzer facts are SSA assumptions in the enclosing sequence.
    # A TileLang metadata region with the old tl.assume spelling carries no
    # native symbolic proof contract.
    body = ir.SeqStmt([*[ir.Evaluate(ir.prim.assume(condition)) for condition in assumptions], body])
    root = sblock("root", body, buffers=[src, dst], annotations={"layout_map": {src: make_ascend_l0c_layout(src)}})
    return tvm.IRModule({"main": tirx.Function(list(params), root)})


def _normalize(mod):
    return tilelang.ascend.transform.NormalizeAscendFractalStorage()(mod)


def _find_copy(mod):
    copies = []

    def collect(node):
        if isinstance(node, ir.Call) and node.op.name in ("tl.tileop.copy", "tl.tileop.ascend_copy"):
            copies.append(node)

    walk_ir(mod["main"].body, collect)
    assert len(copies) == 1
    return copies[0]


def _region_extents(region_call):
    return [axis.extent for axis in region_call.region]


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
    tvm.ir.assert_structural_equal(normalized, _normalize(normalized))


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

    tvm.ir.assert_structural_equal(src_width, dst_width)
    for value in range(5):
        width = substitute(dst_width, {actual_n: ir.IntImm("int32", value)})
        assert int(tvm.sym.Analyzer().simplify(width)) == (8 if value else 0)


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
    for region, expected in zip(copy.args, (src_extents, dst_extents)):
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
    assert int(access_tensor(copy.args[0]).ty.shape[-1]) == 16
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
    tvm.ir.assert_structural_equal(normalized, _normalize(normalized))


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
    tvm.ir.assert_structural_equal(_region_extents(copy.args[1])[-1], n)
    assert "cannot prove padding" in stderr
    assert "undefined" not in stderr


def test_l0c_to_ub_padding_preserves_an_empty_source():
    n = ir.Var("n", "int32")
    mod = _make_fixpipe_module(src_extents=(4, n), params=(n,), assumptions=(n >= 0, n <= 4))
    copy = _find_copy(_normalize(mod))
    width = _region_extents(copy.args[0])[-1]
    for value in (0, 4):
        evaluated = substitute(width, {n: ir.IntImm("int32", value)})
        assert int(tvm.sym.Analyzer().simplify(evaluated)) == (8 if value else 0)


@pytest.mark.parametrize("dual,src_n,dst_n,expected_n", [(False, 4, 4, 8), (False, 10, 4, 8), (True, 10, 8, 16)])
def test_l0c_to_ub_lowered_descriptor_is_aligned(dual, src_n, dst_n, expected_n):
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

    descriptors = []

    @ir.transform.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLowerTileOp":

                def visit(node):
                    if isinstance(node, ir.Call) and node.op.name == "tl.ascend_copy_matrix_cc_to_ub":
                        descriptors.append(tuple(int(node.args[i]) for i in (3, 5, 7)))

                walk_ir(mod["main"].body, visit)

    with tvm.transform.PassContext(instruments=[Capture()]):
        tilelang.lower(main, target="ascend")
    assert descriptors == [(expected_n, 16, int(dual))]


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
    src = tensor((64, 64), dtype, name="src", scope="shared.l1")
    dst = tensor((64, 64), dtype, name="dst", scope="shared.l0a")
    layout = make_ascend_major_mn_layout if major_mn else make_ascend_nz_layout
    before = kernel(
        copy(src, dst, src_shape=source, dst_shape=destination, annotations={"transpose": ir.IntImm("int32", int(transpose))}),
        buffers=[src, dst],
        annotations={"layout_map": {src: layout(src), dst: layout(dst)}},
    )
    after = _normalize(before)
    operation = _find_copy(after)
    assert [int(x) for x in _region_extents(operation.args[0])] == expected_source
    assert [int(x) for x in _region_extents(operation.args[1])] == expected_destination
    tvm.ir.assert_structural_equal(after, _normalize(after))
    with tvm.target.Target("ascend"):
        (load,) = calls(transform.AscendLowerTileOp()(after), "tl.ascend_load_cbuf_to_ca")
    assert int(load.args[7]) == pitch


@pytest.mark.parametrize("empty_operand", ["source", "destination"])
def test_l0_copy_normalization_preserves_dynamic_empty_transfer(empty_operand):
    n = ir.Var("n", "int32")
    rows = ir.prim.Min(ir.prim.Max(n, -1), 24)
    src = tensor((32, 16), "bfloat16", name="src", scope="shared.l1")
    dst = tensor((32, 16), "bfloat16", name="dst", scope="shared.l0a")
    # A fixed source fits the destination whenever its symbolic size is positive.
    src_rows, dst_rows = (rows, 32) if empty_operand == "source" else (1, ir.prim.Min(ir.prim.Max(n, -1), 16))
    before = kernel(
        copy(src, dst, src_shape=[src_rows, 16], dst_shape=[dst_rows, 16]),
        buffers=[src, dst],
        params=[n],
        annotations={"layout_map": {src: make_ascend_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    after = _normalize(before)
    operation = _find_copy(after)
    analyzer = tvm.sym.Analyzer()
    for value in (-1, 0, 1, 24):
        extents = [
            analyzer.simplify(substitute(_region_extents(operand)[0], {n: ir.IntImm("int32", value)})) for operand in operation.args[:2]
        ]
        expected = 0 if value <= 0 else (32 if value == 24 and empty_operand == "source" else 16)
        assert [int(x) for x in extents] == [expected, expected]
    with tvm.target.Target("ascend"):
        lowered = transform.AscendLowerTileOp()(after)
    assert calls(lowered, "tl.ascend_load_cbuf_to_ca")
    (guard,) = nodes(lowered, ir.If)
    for value in (-1, 0, 1, 24):
        condition = analyzer.simplify(substitute(guard.condition, {n: ir.IntImm("int32", value)}))
        assert bool(condition) == (value > 0)


@pytest.mark.parametrize("dtype,source,expected", [("uint8", [24, 3], [32, 4]), ("uint16", [24, 3], [32, 3])])
def test_sf_source_uses_scale_pair_transfer_geometry(dtype, source, expected):
    src = tensor((32, 4), dtype, name="src", scope="shared.l1")
    dst = tensor((32, 4), dtype, name="sf", scope="shared.l0a.sf")
    before = kernel(
        copy(src, dst, src_shape=source, dst_shape=source),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_sf_layout(src)}},
    )
    operation = _find_copy(_normalize(before))
    assert [int(x) for x in _region_extents(operation.args[0])] == expected
    # The destination is a virtual handle describing the bound effective tile.
    assert [int(x) for x in _region_extents(operation.args[1])] == source
