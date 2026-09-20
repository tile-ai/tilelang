"""DMA bounds are clamped without introducing scheduling-time copy guards."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import make_ascend_nz_layout
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
