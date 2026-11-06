"""DMA bounds are clamped without introducing scheduling-time copy guards."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import access_tensor, sblock, lexical_scopes, substitute, tensor

import pytest
import tvm_tirx
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_nz_layout, make_ascend_major_k_layout
from tvm_tirx import tirx
from testing.ascend._ir import calls, nodes


def _copy_module(src_shape, dst_shape, src_min, src_extent, dst_min=None, dst_extent=None, *, scope="shared.l1", transpose=False):
    src = tensor(src_shape, "bfloat16", name="src")
    dst = tensor(dst_shape, "bfloat16", name="dst", scope=scope)

    def region(buffer, mask, mins, extents):
        return ir.TensorRegion(
            buffer, [ir.Range.from_min_extent(start, extent) for start, extent in zip(mins, extents)], ir.TensorRegionType()
        )

    copy = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"),
        tvm_tirx.ir.Op.get("tl.tileop.ascend_copy"),
        [region(src, 1, src_min, src_extent), region(dst, 2, dst_min or [0] * len(dst_shape), dst_extent or dst_shape)],
        {"transpose": ir.IntImm("int32", 1)} if transpose else {},
        ir.UnknownLoc(),
    )
    annotations = {"layout_map": {dst: make_ascend_nz_layout(dst)}} if scope == "shared.l1" else {}
    root = sblock("root", ir.Evaluate(copy), buffers=[dst], annotations=annotations)
    body = root
    params = [src, *tirx.analysis.undefined_vars(body, [src])]
    return tvm_tirx.IRModule({"main": tirx.Function(params, body)})


@pytest.mark.parametrize("transpose", [False, True], ids=["direct", "transpose"])
def test_clamped_copy_and_padding_use_destination_axes(transpose):
    before = _copy_module((8, 10) if transpose else (10, 8), (16, 32), [0, 0], [32, 16] if transpose else [16, 32], transpose=transpose)
    after = transform.AscendInsertOOBPadding()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    assert [int(axis.extent) for axis in copy.args[0].region] == ([8, 10] if transpose else [10, 8])
    assert [int(axis.extent) for axis in copy.args[1].region] == [10, 8]
    fills = calls(after, "tl.tileop.fill")
    regions = {
        tuple(
            int(tvm_tirx.sym.Analyzer().simplify(x))
            for x in [*[axis.begin for axis in call.args[0].region], *[axis.extent for axis in call.args[0].region]]
        )
        for call in fills
    }
    assert regions == {(0, 16, 16, 16), (10, 0, 6, 32)}
    assert all(float(call.args[1]) == 0 for call in fills)
    assert not nodes(after, ir.If)


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
        with pytest.raises(RuntimeError, match=message):
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
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_tile_aligned_opaque_offset_has_no_partial_row_padding():
    index = tensor((1,), "int32", scope="local.var")[0]
    before = _copy_module((64, 16), (16, 16), [index * 16, 0], [16, 16])
    after = transform.AscendInsertOOBPadding()(before)
    assert not calls(after, "tl.tileop.fill")


def test_dynamic_copy_clamps_extents_without_guarding_the_first_write():
    rows, group = ir.Var("rows", "int32"), ir.Var("group", "int32")
    before = _copy_module((4, 16, 64), (16, 64), [group, 0, 0], [1, rows, 64], dst_extent=[rows, 64], scope="shared.dyn")
    after = transform.AscendInsertOOBPadding()(before)
    assert not nodes(after, ir.If)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    # Native allocation/metadata definitions precede the same unguarded first
    # write directly in the root; no intervening loop or region may guard it.
    (root,) = lexical_scopes(after)
    assert isinstance(root.body, ir.SeqStmt)
    assert isinstance(root.body.seq[-1], ir.Evaluate) and root.body.seq[-1].value.same_as(copy)
    assert all(
        (isinstance(statement, ir.Bind) and statement.value.op.name in {"tirx.alloc_tensor", "tirx.decl_tensor"})
        or (isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call) and statement.value.op.name == "tl.annotation")
        for statement in root.body.seq[:-1]
    )
    for group_value, row_count in [(0, 0), (0, 5), (3, 16), (4, 5)]:
        actual = [
            int(tvm_tirx.sym.Analyzer().simplify(substitute(x, {rows: row_count, group: group_value})))
            for x in [axis.extent for axis in copy.args[0].region]
        ]
        assert actual == [int(group_value < 4), row_count, 64]


def _mx_operand_copy(layout, actual_k, k_alloc=128, dtype="float8_e4m3fn"):
    mn = 64 if dtype == "float4_e2m1fn" else 32
    src_transposed = layout != "k-contiguous"
    dst_transposed = layout == "mn-contiguous"
    src_shape = (k_alloc, mn) if src_transposed else (mn, k_alloc)
    dst_shape = (k_alloc, mn) if dst_transposed else (mn, k_alloc)
    a = tensor(src_shape, dtype, name="A")
    l1 = tensor(dst_shape, dtype, name="l1", scope="shared.l1")
    scale = tensor((mn, k_alloc // 64), "int16", name="scale", scope="shared.l1")
    l0a = tensor(dst_shape, dtype, name="l0a", scope="shared.l0a")
    l0b = tensor((mn, k_alloc), dtype, name="l0b", scope="shared.l0b")
    sfa = tensor((mn, k_alloc // 64), "int16", name="sfa", scope="shared.l0a.sf")
    sfb = tensor((mn, k_alloc // 64), "int16", name="sfb", scope="shared.l0b.sf")
    acc = tensor((mn, mn), "float32", name="acc", scope="shared.l0c")

    def region(buffer, transposed):
        shape = (actual_k, mn) if transposed else (mn, actual_k)
        return ir.TensorRegion(buffer, [tvm_tirx.ir.Range.from_min_extent(0, extent) for extent in shape], ir.TensorRegionType())

    # The MX load and consumer identify the semantic K axis. Their other inputs
    # are already available at this pass boundary; no full kernel is needed.
    body = ir.SeqStmt(
        [
            ir.Evaluate(T.copy(region(a, src_transposed), region(l1, dst_transposed), transpose=src_transposed != dst_transposed)),
            ir.Evaluate(T.copy(l1, l0a)),
            ir.Evaluate(T.copy(scale, sfa)),
            ir.Evaluate(T.gemm_blockscaled(l0a, l0b, acc, sfa, sfb, transpose_A=dst_transposed, transpose_B=True, clear_accum=True)),
        ]
    )
    root = sblock(
        "root",
        body,
        buffers=[l1, scale, l0a, l0b, acc, sfa, sfb],
        annotations={"tl.l0_sf_bindings": {sfa: l0a, sfb: l0b}, "layout_map": {l1: make_ascend_major_k_layout(l1)}},
    )
    return tvm_tirx.IRModule({"main": tirx.Function([a], root)}), l1


@pytest.mark.parametrize(
    "dtype, layout, actual_k, tail",
    [
        ("float8_e4m3fn", "k-contiguous", 65, (96, 32)),
        ("float8_e4m3fn", "mn-contiguous", 65, (65, 63)),
        ("float8_e4m3fn", "transpose-to-k-contiguous", 65, (96, 32)),
        ("float8_e4m3fn", "k-contiguous", 33, None),
        ("float8_e4m3fn", "mn-contiguous", 64, None),
        ("float8_e4m3fn", "transpose-to-k-contiguous", 33, None),
        ("float4_e2m1fn", "k-contiguous", 4, None),
        ("float4_e2m1fn", "k-contiguous", 34, None),
        ("float4_e2m1fn", "mn-contiguous", 4, (4, 60)),
        ("float4_e2m1fn", "mn-contiguous", 34, (34, 30)),
        ("float4_e2m1fn", "mn-contiguous", 66, (66, 62)),
        ("float4_e2m1fn", "mn-contiguous", 64, None),
        ("float4_e2m1fn", "transpose-to-k-contiguous", 66, None),
    ],
)
def test_mx_copy_pads_only_the_unwritten_k_tail(dtype, layout, actual_k, tail):
    before, l1 = _mx_operand_copy(layout, actual_k, dtype=dtype)
    after = transform.AscendInsertOOBPadding()(before)
    fills = calls(after, "tl.tileop.fill")
    assert len(fills) == (tail is not None)
    if tail is None:
        return
    (fill,) = fills
    dst, value = fill.args[:2]
    assert access_tensor(dst).same_as(l1)
    assert float(value) == 0
    k_axis = 0 if layout == "mn-contiguous" else 1
    start, extent = tail
    assert [int(axis.begin) for axis in dst.region] == ([start, 0] if k_axis == 0 else [0, start])
    mn = 64 if dtype == "float4_e2m1fn" else 32
    assert [int(axis.extent) for axis in dst.region] == ([extent, mn] if k_axis == 0 else [mn, extent])
    # Padding must follow the GM write and precede the MX operand's L1 read.
    copies = calls(after, "tl.tileop.ascend_copy")
    operations = nodes(after, ir.Call)

    def position(call):
        return next(i for i, candidate in enumerate(operations) if candidate.same_as(call))

    assert position(copies[0]) < position(fill) < position(copies[1])


def test_mx_padding_rejects_an_incomplete_physical_scale_group():
    before, _ = _mx_operand_copy("k-contiguous", 96, k_alloc=96)
    with pytest.raises(ValueError, match="divisible by 64"):
        transform.AscendInsertOOBPadding()(before)
