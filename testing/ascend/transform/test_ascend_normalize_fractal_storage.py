"""Canonical Cube allocations, aliases and logical-to-padded indices."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0c_layout, make_ascend_nz_layout
from tvm import tirx
from testing.ascend._ir import calls, nodes


def _module(owner, view, body):
    block = tirx.SBlock([], [], [], "root", body, alloc_buffers=[owner], annotations={"layout_map": {view: make_ascend_nz_layout(view)}})
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, block))})


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
    buffer = tirx.decl_buffer(shape, dtype, name="tile", scope="shared.l1")
    before = _module(buffer, buffer, tirx.Evaluate(buffer[tuple(0 for _ in shape)]))
    after = transform.NormalizeAscendFractalStorage()(before)
    allocation = after["main"].body.block.alloc_buffers[0]
    assert tuple(int(x) for x in allocation.shape) == padded
    assert allocation.data.same_as(buffer.data)
    tvm.ir.assert_structural_equal(after, transform.NormalizeAscendFractalStorage()(after))


@pytest.mark.parametrize("kind", ["view", "reshape", "reinterpret"])
def test_alias_accesses_use_one_canonical_allocation(kind):
    owner_shape = (15, 15) if kind == "view" else (225,)
    owner = tirx.decl_buffer(owner_shape, "uint16" if kind == "reinterpret" else "bfloat16", name="owner", scope="shared.l1")
    view = tirx.decl_buffer((15, 15), "bfloat16", data=owner.data)
    before = _module(owner, view, tirx.BufferStore(view, tirx.const(1, "bfloat16"), [1, 0]))
    after = transform.NormalizeAscendFractalStorage()(before)
    allocation = after["main"].body.block.alloc_buffers[0]
    (store,) = nodes(after, tirx.BufferStore)
    assert store.buffer.same_as(allocation)
    assert allocation.data.same_as(owner.data)
    assert allocation.dtype == "bfloat16"
    assert tuple(int(x) for x in allocation.shape) == (16, 16)
    assert tuple(int(x) for x in store.indices) == (1, 0)


@pytest.mark.parametrize("raw", [False, True], ids=["reshaped-access", "raw-pointer"])
def test_pointer_base_is_reexpressed_in_canonical_indices(raw):
    owner = tirx.decl_buffer((15, 15), "bfloat16", name="owner", scope="shared.l1")
    alias = tirx.decl_buffer((3, 75), "bfloat16", data=owner.data)
    row = tirx.Var("row", "int32")
    pointer = owner.access_ptr("r", offset=15, extent=1) if raw else T.access_ptr(alias[row, 0], "r", extent=1)
    body = tirx.Evaluate(tirx.call_extern("int32", "consume", pointer))
    before = _module(owner, owner, body if raw else tirx.For(row, 0, 3, tirx.ForKind.SERIAL, body))
    after = transform.NormalizeAscendFractalStorage()(before)
    (pointer,) = calls(after, "tl.access_ptr")
    load, extent, _ = pointer.args
    assert load.buffer.same_as(after["main"].body.block.alloc_buffers[0])
    analyzer = tvm.arith.Analyzer()
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
    owner = tirx.decl_buffer((450 if kind == "width" else 225,), "uint8" if kind == "width" else "uint16", scope="shared.l1")
    view = tirx.decl_buffer((15, 15), "bfloat16", data=owner.data)
    body = tirx.Evaluate(view[0, 0])
    if kind == "raw-dtype":
        body = tirx.SeqStmt([body, tirx.Evaluate(tirx.call_extern("int32", "consume", owner.access_ptr("r", offset=15, extent=1)))])
    before = _module(owner, view, body)
    with pytest.raises(tvm.error.InternalError, match=message):
        transform.NormalizeAscendFractalStorage()(before)


@pytest.mark.parametrize("rows,expected_rows", [(17, 17), (31, 32)])
def test_oob_copy_expands_only_a_complete_destination_axis(rows, expected_rows):
    src = tirx.decl_buffer((31, 128), "bfloat16")
    dst = tirx.decl_buffer((31, 64), "bfloat16", scope="shared.l1")
    read = tirx.BufferRegion(src, [tvm.ir.Range(0, rows), tvm.ir.Range(112, 176)])
    write = tirx.BufferRegion(dst, [tvm.ir.Range(0, rows), tvm.ir.Range(0, 64)])
    before = _module(dst, dst, tirx.Evaluate(T.copy(read, write)))
    after = transform.NormalizeAscendFractalStorage()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    assert [int(x) for x in copy.args[0].args[2:]] == [rows, 64]
    assert [int(x) for x in copy.args[1].args[2:]] == [expected_rows, 64]


def _make_alias_region_module(view_shape, mins, extents):
    owner = tirx.decl_buffer((15, 15), "bfloat16", name="owner", scope="shared.l1")
    view = tirx.decl_buffer(view_shape, "bfloat16", name="view", scope="shared.l1", data=owner.data)
    layout = tilelang.layout.make_ascend_nz_layout(owner)
    ranges = [tvm.ir.Range.from_min_extent(start, extent) for start, extent in zip(mins, extents)]
    region = tirx.BufferRegion(view, ranges)
    load = tirx.BufferLoad(view, mins)
    region_call = tirx.Call("handle", tvm.ir.Op.get("tl.region"), [load, tirx.const(1), *[tirx.const(x) for x in extents]])
    body = tirx.SeqStmt(
        [tirx.DeclBuffer(view), tirx.BufferStore(view, tirx.const(7, "bfloat16"), mins), tirx.Evaluate(load), tirx.Evaluate(region_call)]
    )
    block = tirx.SBlock(
        [],
        [region],
        [region],
        "root",
        body,
        alloc_buffers=[owner],
        annotations={"layout_map": {owner: layout, view: layout.reshape(view_shape)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, block))})


def test_alias_indices_and_access_regions_use_the_same_padded_storage():
    normalized = tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((225,), (15,), (30,)))
    root = normalized["main"].body.block
    owner = root.alloc_buffers[0]
    alias = root.reads[0].buffer
    assert owner.same_as(alias)
    assert owner.data.same_as(alias.data)
    assert tuple(int(x) for x in owner.shape) == tuple(int(x) for x in alias.shape) == (16, 16)
    assert [(int(r.min), int(r.extent)) for r in root.reads[0].region] == [(1, 2), (0, 15)]
    tvm.ir.assert_structural_equal(root.reads[0], root.writes[0])
    for node in [root.body.seq[1], root.body.seq[2].value, root.body.seq[3].value.args[0]]:
        assert tuple(int(x) for x in node.indices) == (1, 0)
        assert tuple(int(x) for x in node.buffer.shape) == (16, 16)
    tvm.ir.assert_structural_equal(normalized, tilelang.ascend.transform.NormalizeAscendFractalStorage()(normalized))


def test_alias_normalization_preserves_allocation_properties():
    owner = tirx.decl_buffer((225,), "uint16", name="owner", scope="shared.l1", data_alignment=1024, offset_factor=8, elem_offset=0)
    view = tirx.decl_buffer((15, 15), "bfloat16", name="view", scope="shared.l1", data=owner.data)
    layout = tilelang.layout.make_ascend_nz_layout(view)
    block = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.Evaluate(tirx.BufferLoad(view, [1, 0])),
        alloc_buffers=[owner],
        annotations={"layout_map": {view: layout, owner: layout.reshape((225,))}},
    )
    mod = tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, block))})
    normalized = tilelang.ascend.transform.NormalizeAscendFractalStorage()(mod)
    allocation = normalized["main"].body.block.alloc_buffers[0]
    assert allocation.data.same_as(owner.data)
    assert allocation.data_alignment == 1024
    assert allocation.offset_factor == 8
    assert allocation.dtype == "bfloat16"
    assert tuple(int(x) for x in allocation.shape) == (16, 16)
    assert allocation.same_as(normalized["main"].body.block.body.value.buffer)


def test_non_rectangular_alias_region_is_rejected():
    # Three logical elements in different rows cannot become an 11-row DMA.
    with pytest.raises(tvm.error.InternalError, match="not a contiguous rectangular matrix"):
        tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((3, 75), (0, 0), (3, 1)))


def test_overlapping_alias_region_is_rejected():
    # The 3x80 source box repeats logical elements across the 75-element row
    # stride, even though its bounding box is also 240 elements (16x15).
    with pytest.raises(tvm.error.InternalError, match="crosses an original reshape axis"):
        tilelang.ascend.transform.NormalizeAscendFractalStorage()(_make_alias_region_module((3, 75), (0, 0), (3, 80)))


def _make_region(buffer, extents, access_mask, mins=None):
    return tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.region"),
        [
            tirx.BufferLoad(buffer, mins if mins is not None else [0] * len(extents)),
            tirx.const(access_mask),
            *extents,
        ],
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
    src = tirx.decl_buffer((16, src_cols), "float32", name="l0c", scope="shared.l0c")
    dst = tirx.decl_buffer(
        dst_shape if dst_shape is not None else (16, dst_cols),
        dst_dtype,
        name="ub",
        scope="shared",
        strides=dst_strides,
    )
    annotations = tvm.runtime.convert({"dual_dst_ctl": tirx.IntImm("int32", dual_dst_ctl)})
    copy = tirx.Call(
        "handle",
        tvm.ir.Op.get(copy_op),
        [
            _make_region(src, src_extents if src_extents is not None else (rows, cols), 1),
            _make_region(dst, dst_extents if dst_extents is not None else (rows, cols), 2, dst_mins),
        ],
        annotations,
    )
    body = tirx.Evaluate(copy)
    for condition in reversed(assumptions):
        body = tirx.AttrStmt(
            condition,
            "tl.assume",
            tirx.StringImm(f"Assume: {condition}"),
            body,
        )
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_l0c_layout(src)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc(list(params), tirx.SBlockRealize([], True, root))})


def _normalize(mod):
    return tilelang.ascend.transform.NormalizeAscendFractalStorage()(mod)


def _find_copy(mod):
    copies = []

    def collect(node):
        if isinstance(node, tirx.Call) and node.op.name in ("tl.tileop.copy", "tl.tileop.ascend_copy"):
            copies.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, collect)
    assert len(copies) == 1
    return copies[0]


def _region_extents(region_call):
    return list(region_call.args[2:])


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
    actual_n = tirx.Var("actual_n", "int32")
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
        width = tirx.stmt_functor.substitute(dst_width, {actual_n: tirx.IntImm("int32", value)})
        assert int(tvm.arith.Analyzer().simplify(width)) == (8 if value else 0)


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
    n = tirx.Var("n", "int32")
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
    assert int(copy.args[0].args[0].buffer.shape[-1]) == 16
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
    stride = tirx.Var("inner_stride", "int32")
    mod = _make_fixpipe_module(dst_strides=(8, stride), params=(stride,), assumptions=(stride == 1,) if known_unit_stride else ())
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err
    assert int(_region_extents(copy.args[1])[-1]) == (8 if known_unit_stride else 4)
    assert ("cannot prove a contiguous 2D destination layout" in stderr) == (not known_unit_stride)


@pytest.mark.parametrize("known_aligned_stride", [False, True])
def test_l0c_to_ub_symbolic_row_stride(capfd, known_aligned_stride):
    stride = tirx.Var("row_stride", "int32")
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
    assert int(copy.args[1].args[0].indices[-1]) == col_min


def test_l0c_to_ub_trailing_unit_axis_uses_the_transfer_axis():
    mod = _make_fixpipe_module(dst_shape=(16, 16, 1), dst_extents=(4, 4, 1))
    copy = _find_copy(_normalize(mod))
    assert [int(x) for x in _region_extents(copy.args[0])] == [4, 8]
    assert [int(x) for x in _region_extents(copy.args[1])] == [4, 8, 1]


def test_l0c_to_ub_unknown_capacity_keeps_the_original_region(capfd):
    n = tirx.Var("n", "int32")
    mod = _make_fixpipe_module(cols=n, params=(n,))
    capfd.readouterr()
    copy = _find_copy(_normalize(mod))
    stderr = capfd.readouterr().err
    tvm.ir.assert_structural_equal(_region_extents(copy.args[1])[-1], n)
    assert "cannot prove padding" in stderr
    assert "undefined" not in stderr


def test_l0c_to_ub_padding_preserves_an_empty_source():
    n = tirx.Var("n", "int32")
    mod = _make_fixpipe_module(src_extents=(4, n), params=(n,), assumptions=(n >= 0, n <= 4))
    copy = _find_copy(_normalize(mod))
    width = _region_extents(copy.args[0])[-1]
    for value in (0, 4):
        evaluated = tirx.stmt_functor.substitute(width, {n: tirx.IntImm("int32", value)})
        assert int(tvm.arith.Analyzer().simplify(evaluated)) == (8 if value else 0)


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

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLowerTileOp":

                def visit(node):
                    if isinstance(node, tirx.Call) and node.op.name == "tl.ascend_copy_matrix_cc_to_ub":
                        descriptors.append(tuple(int(node.args[i]) for i in (3, 5, 7)))

                tirx.stmt_functor.post_order_visit(mod["main"].body, visit)

    with tvm.transform.PassContext(instruments=[Capture()]):
        tilelang.lower(main, target="ascend")
    assert descriptors == [(expected_n, 16, int(dual))]
