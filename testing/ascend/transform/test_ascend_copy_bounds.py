"""DMA bounds are clamped without introducing scheduling-time copy guards."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_nz_layout, make_ascend_major_k_layout
from tvm import tirx
from testing.ascend._ir import calls, nodes


def _copy_module(src_shape, dst_shape, src_min, src_extent, dst_min=None, dst_extent=None, *, scope="shared.l1", transpose=False):
    src = tirx.decl_buffer(src_shape, "bfloat16", name="src")
    dst = tirx.decl_buffer(dst_shape, "bfloat16", name="dst", scope=scope)

    def region(buffer, mask, mins, extents):
        return tirx.Call("handle", tvm.ir.Op.get("tl.region"), [buffer[tuple(mins)], mask, *extents])

    copy = tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.ascend_copy"),
        [
            region(src, 1, src_min, src_extent),
            region(dst, 2, dst_min or [0] * len(dst_shape), dst_extent or dst_shape),
        ],
        annotations={"transpose": tirx.IntImm("int32", 1)} if transpose else {},
    )
    annotations = {"layout_map": {dst: make_ascend_nz_layout(dst)}} if scope == "shared.l1" else {}
    root = tirx.SBlock([], [], [], "root", tirx.Evaluate(copy), alloc_buffers=[dst], annotations=annotations)
    body = tirx.SBlockRealize([], True, root)
    params = [src.data, *tirx.analysis.undefined_vars(body, [src.data])]
    return tvm.IRModule({"main": tirx.PrimFunc(params, body, buffer_map={src.data: src})})


@pytest.mark.parametrize("transpose", [False, True], ids=["direct", "transpose"])
def test_clamped_copy_and_padding_use_destination_axes(transpose):
    before = _copy_module((8, 10) if transpose else (10, 8), (16, 32), [0, 0], [32, 16] if transpose else [16, 32], transpose=transpose)
    after = transform.AscendInsertOOBPadding()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    assert [int(x) for x in copy.args[0].args[2:]] == ([8, 10] if transpose else [10, 8])
    assert [int(x) for x in copy.args[1].args[2:]] == [10, 8]
    fills = calls(after, "tl.tileop.fill")
    regions = {
        tuple(int(tvm.arith.Analyzer().simplify(x)) for x in [*call.args[0].args[0].indices, *call.args[0].args[2:]]) for call in fills
    }
    assert regions == {(0, 16, 16, 16), (10, 0, 6, 32)}
    assert all(float(call.args[1]) == 0 for call in fills)
    assert not nodes(after, tirx.IfThenElse)


@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize("complete", [False, True], ids=["partial-axis", "full-axis"])
def test_padding_requires_the_complete_orthogonal_axis(axis, complete):
    width = 32 if complete else 17
    if axis == "row":
        before = _copy_module((16, 64), (16, 32), [8, 0], [16, width], dst_extent=[16, width])
        message = "covering the full col axis"
    else:
        before = _copy_module((32, 128), (32, 64), [0, 112], [width, 64], dst_extent=[width, 64])
        message = "covering the full row axis"
    if not complete:
        with pytest.raises(tvm.error.InternalError, match=message):
            transform.AscendInsertOOBPadding()(before)
    else:
        after = transform.AscendInsertOOBPadding()(before)
        assert len(calls(after, "tl.tileop.fill")) == 1


@pytest.mark.parametrize("case", ["short-region", "larger-destination", "padded-destination-row"])
def test_in_bounds_copy_does_not_fill_extra_storage(case):
    if case == "short-region":
        before = _copy_module((16, 64), (16, 16), [0, 0], [15, 15], dst_extent=[15, 15])
    elif case == "larger-destination":
        before = _copy_module((16, 32), (16, 64), [0, 0], [16, 32])
    else:
        before = _copy_module((1, 32), (16, 32), [0, 0], [1, 32], dst_min=[15, 0], dst_extent=[1, 32])
    after = transform.AscendInsertOOBPadding()(before)
    tvm.ir.assert_structural_equal(after, before)


def test_tile_aligned_opaque_offset_has_no_partial_row_padding():
    index = tirx.decl_buffer((1,), "int32", scope="local.var")[0]
    before = _copy_module((64, 16), (16, 16), [index * 16, 0], [16, 16])
    after = transform.AscendInsertOOBPadding()(before)
    assert not calls(after, "tl.tileop.fill")


def test_dynamic_copy_clamps_extents_without_guarding_the_first_write():
    rows, group = tirx.Var("rows", "int32"), tirx.Var("group", "int32")
    before = _copy_module((4, 16, 64), (16, 64), [group, 0, 0], [1, rows, 64], dst_extent=[rows, 64], scope="shared.dyn")
    after = transform.AscendInsertOOBPadding()(before)
    assert not nodes(after, tirx.IfThenElse)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    assert isinstance(after["main"].body.block.body, tirx.Evaluate)
    for group_value, row_count in [(0, 0), (0, 5), (3, 16), (4, 5)]:
        actual = [
            int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(x, {rows: row_count, group: group_value})))
            for x in copy.args[0].args[2:]
        ]
        assert actual == [int(group_value < 4), row_count, 64]


def _mx_operand_copy(layout, actual_k, k_alloc=128):
    mn = 32
    src_transposed = layout != "k-contiguous"
    dst_transposed = layout == "mn-contiguous"
    src_shape = (k_alloc, mn) if src_transposed else (mn, k_alloc)
    dst_shape = (k_alloc, mn) if dst_transposed else (mn, k_alloc)
    a = tirx.decl_buffer(src_shape, "float8_e4m3fn", name="A")
    l1 = tirx.decl_buffer(dst_shape, "float8_e4m3fn", name="l1", scope="shared.l1")
    scale = tirx.decl_buffer((mn, k_alloc // 64), "int16", name="scale", scope="shared.l1")
    l0a = tirx.decl_buffer(dst_shape, "float8_e4m3fn", name="l0a", scope="shared.l0a")
    l0b = tirx.decl_buffer((mn, k_alloc), "float8_e4m3fn", name="l0b", scope="shared.l0b")
    sfa = tirx.decl_buffer((mn, k_alloc // 64), "int16", name="sfa", scope="shared.l0a.sf")
    sfb = tirx.decl_buffer((mn, k_alloc // 64), "int16", name="sfb", scope="shared.l0b.sf")
    acc = tirx.decl_buffer((mn, mn), "float32", name="acc", scope="shared.l0c")

    def region(buffer, transposed):
        shape = (actual_k, mn) if transposed else (mn, actual_k)
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, extent) for extent in shape])

    # The MX load and consumer identify the semantic K axis. Their other inputs
    # are already available at this pass boundary; no full kernel is needed.
    body = tirx.SeqStmt(
        [
            tirx.Evaluate(T.copy(region(a, src_transposed), region(l1, dst_transposed), transpose=src_transposed != dst_transposed)),
            tirx.Evaluate(T.copy(l1, l0a)),
            tirx.Evaluate(T.copy(scale, sfa)),
            tirx.Evaluate(T.gemm_blockscaled(l0a, l0b, acc, sfa, sfb, transpose_A=dst_transposed, transpose_B=True, clear_accum=True)),
        ]
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[l1, scale, l0a, l0b, acc, sfa, sfb],
        annotations={"tl.l0_sf_bindings": {sfa.data: l0a.data, sfb.data: l0b.data}, "layout_map": {l1: make_ascend_major_k_layout(l1)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc([a.data], tirx.SBlockRealize([], True, root), buffer_map={a.data: a})}), l1


@pytest.mark.parametrize(
    "layout, actual_k, tail",
    [
        ("k-contiguous", 65, (96, 32)),
        ("mn-contiguous", 65, (65, 63)),
        ("transpose-to-k-contiguous", 65, (96, 32)),
        ("k-contiguous", 33, None),
        ("mn-contiguous", 64, None),
        ("transpose-to-k-contiguous", 33, None),
    ],
)
def test_mx_copy_pads_only_the_unwritten_k_tail(layout, actual_k, tail):
    before, l1 = _mx_operand_copy(layout, actual_k)
    after = transform.AscendInsertOOBPadding()(before)
    fills = calls(after, "tl.tileop.fill")
    assert len(fills) == (tail is not None)
    if tail is None:
        return
    (fill,) = fills
    dst, value = fill.args
    assert dst.args[0].buffer.same_as(l1)
    assert float(value) == 0
    k_axis = 0 if layout == "mn-contiguous" else 1
    start, extent = tail
    assert [int(index) for index in dst.args[0].indices] == ([start, 0] if k_axis == 0 else [0, start])
    assert [int(size) for size in dst.args[2:]] == ([extent, 32] if k_axis == 0 else [32, extent])
    # Padding must follow the GM write and precede the MX operand's L1 read.
    copies = calls(after, "tl.tileop.ascend_copy")
    operations = nodes(after, tirx.Call)

    def position(call):
        return next(i for i, candidate in enumerate(operations) if candidate.same_as(call))

    assert position(copies[0]) < position(fill) < position(copies[1])


def test_mx_padding_rejects_an_incomplete_physical_scale_group():
    before, _ = _mx_operand_copy("k-contiguous", 96, k_alloc=96)
    with pytest.raises(tvm.error.InternalError, match="divisible by 64"):
        transform.AscendInsertOOBPadding()(before)
