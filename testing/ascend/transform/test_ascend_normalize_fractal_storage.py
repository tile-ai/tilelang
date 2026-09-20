"""Canonical Cube allocations, aliases and logical-to-padded indices."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_nz_layout
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
