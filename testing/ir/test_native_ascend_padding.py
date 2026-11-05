"""Native pre-schedule padding preserves semantic regions and fill ordering."""

import os
from pathlib import Path

import pytest
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.ascend import language as T
from tilelang.layout import make_ascend_major_k_layout

from testing.ascend._ir import substitute

from test_native_ascend_ops import constant, emit_source, function, subregion
from test_native_cuda_copy import calls
from test_native_reducer_contracts import allocate, annotation, attribute, decoded, loop, objects, region, run_pass, tensor, tileop


def pad(value, name):
    original = ir.save_json(value)
    result = run_pass("AscendInsertOOBPadding", value)
    assert original == ir.save_json(value)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    if directory := os.environ.get("TILELANG_OPS_ARTIFACT_DIR"):
        (Path(directory) / (name + ".json")).write_text(ir.save_json(result))
    return result


def copy_case(
    src_shape,
    dst_shape,
    starts,
    extents,
    dst_starts=None,
    dst_extents=None,
    *,
    scope="shared.l1",
    transpose=False,
    parameters=(),
    carrier="annotation",
):
    src = tensor("src", src_shape, "bfloat16", "global")
    dst = tensor("dst", dst_shape, "bfloat16", scope)
    copy = tileop(
        "ascend_copy",
        subregion(src, starts, extents),
        subregion(dst, dst_starts or [0] * len(dst_shape), dst_extents or dst_shape),
        metadata={"transpose": 1} if transpose else {},
    )
    metadata = {"layout_map": {dst: make_ascend_major_k_layout(dst)}} if scope == "shared.l1" else {}
    if carrier == "annotation":
        body = [annotation(metadata), copy] if metadata else [copy]
    elif carrier == "attribute":
        body = [attribute(None, "layout_map", metadata["layout_map"], [copy])]
    else:
        body = [attribute(None, "tl.kernel", metadata, [copy])]
    return function([src, *parameters], [allocate(dst), *body])


def range_values(value):
    return [constant(r.begin) for r in value.region] + [constant(r.extent) for r in value.region]


@pytest.mark.parametrize("transpose", [False, True])
def test_clamped_copy_and_padding_use_destination_axes(transpose, tmp_path):
    value = copy_case((8, 10) if transpose else (10, 8), (16, 32), [0, 0], [32, 16] if transpose else [16, 32], transpose=transpose)
    result = pad(value, "oob-transpose-" + str(transpose))
    (copy,) = calls(result, "tl.tileop.ascend_copy")
    operands, metadata = decoded(copy)
    assert [constant(r.extent) for r in operands[0].region] == ([8, 10] if transpose else [10, 8])
    assert [constant(r.extent) for r in operands[1].region] == [10, 8]
    assert int(metadata.get("transpose", 0)) == int(transpose)
    fills = calls(result, "tl.tileop.fill")
    assert {tuple(range_values(decoded(fill)[0][0])) for fill in fills} == {(0, 16, 16, 16), (10, 0, 6, 32)}
    assert all(float(decoded(fill)[0][1].value) == 0 for fill in fills)
    assert not objects(result, ir.If)
    lowered = run_pass("AscendLowerTileOp", run_pass("LayoutInference", result))
    code = emit_source(lowered, tmp_path)
    assert "asc_fill_l1" in code and "asc_copy_gm2l1" in code


@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize("complete", [False, True])
def test_padding_requires_the_complete_orthogonal_axis(axis, complete):
    width = 32 if complete else 17
    if axis == "row":
        value = copy_case((16, 64), (16, 32), [8, 0], [16, width], dst_extents=[16, width])
        diagnostic = "covering the full col axis"
    else:
        value = copy_case((32, 128), (32, 64), [0, 112], [width, 64], dst_extents=[width, 64])
        diagnostic = "covering the full row axis"
    if complete:
        assert len(calls(pad(value, "oob-full-" + axis), "tl.tileop.fill")) == 1
    else:
        before = ir.save_json(value)
        with pytest.raises(Exception, match=diagnostic):
            run_pass("AscendInsertOOBPadding", value)
        assert ir.save_json(value) == before


@pytest.mark.parametrize("case", ["short-region", "larger-destination", "padded-row"])
def test_in_bounds_copy_does_not_fill_extra_storage(case):
    if case == "short-region":
        value = copy_case((16, 64), (16, 16), [0, 0], [15, 15], dst_extents=[15, 15])
    elif case == "larger-destination":
        value = copy_case((16, 32), (16, 64), [0, 0], [16, 32])
    else:
        value = copy_case((1, 32), (16, 32), [0, 0], [1, 32], dst_starts=[15, 0], dst_extents=[1, 32])
    ir.assert_structural_equal(pad(value, "oob-inbounds-" + case), value)


def test_tile_aligned_opaque_offset_has_no_partial_row_padding():
    index = tensor("index", [1], "int32", "global")
    value = copy_case((64, 16), (16, 16), [ir.TensorLoad(index, [0]) * 16, 0], [16, 16], parameters=[index])
    assert not calls(pad(value, "oob-opaque-index"), "tl.tileop.fill")


def test_dynamic_copy_clamps_extents_without_guarding_the_first_write():
    rows, group = ir.Var("rows", "int32"), ir.Var("group", "int32")
    value = copy_case(
        (4, 16, 64), (16, 64), [group, 0, 0], [1, rows, 64], dst_extents=[rows, 64], scope="shared.dyn", parameters=[rows, group]
    )
    result = pad(value, "oob-dynamic-copy")
    assert not objects(result, ir.If)
    (copy,) = calls(result, "tl.tileop.ascend_copy")
    source = decoded(copy)[0][0]
    for group_value, row_count in [(0, 0), (0, 5), (3, 16), (4, 5)]:
        analyzer = tvm_tirx.sym.Analyzer()
        values = {rows: row_count, group: group_value}
        assert [int(analyzer.simplify(substitute(r.extent, values))) for r in source.region] == [int(group_value < 4), row_count, 64]


def mx_case(layout, actual_k, *, k_alloc=128, dtype="float8_e4m3fn", suffix="", parameters=()):
    mn = 64 if dtype == "float4_e2m1fn" else 32
    src_transpose, dst_transpose = layout != "k-contiguous", layout == "mn-contiguous"
    src_shape = (k_alloc, mn) if src_transpose else (mn, k_alloc)
    dst_shape = (k_alloc, mn) if dst_transpose else (mn, k_alloc)
    gm = tensor("gm" + suffix, src_shape, dtype, "global")
    l1 = tensor("l1" + suffix, dst_shape, dtype, "shared.l1")
    scale = tensor("scale" + suffix, (mn, k_alloc // 64), "int16", "shared.l1")
    a = tensor("a" + suffix, dst_shape, dtype, "shared.l0a")
    b = tensor("b" + suffix, (mn, k_alloc), dtype, "shared.l0b")
    sa = tensor("sa" + suffix, (mn, k_alloc // 64), "int16", "shared.l0a.sf")
    sb = tensor("sb" + suffix, (mn, k_alloc // 64), "int16", "shared.l0b.sf")
    c = tensor("c" + suffix, (mn, mn), "float32", "shared.l0c")
    source = subregion(gm, [0, 0], [actual_k, mn] if src_transpose else [mn, actual_k])
    destination = subregion(l1, [0, 0], [actual_k, mn] if dst_transpose else [mn, actual_k])
    metadata = {"layout_map": {l1: make_ascend_major_k_layout(l1)}, "tl.l0_sf_bindings": {sa: a, sb: b}}
    operations = [
        ir.Evaluate(T.copy(source, destination, transpose=src_transpose != dst_transpose)),
        ir.Evaluate(T.copy(l1, a)),
        ir.Evaluate(T.copy(scale, sa)),
        ir.Evaluate(T.gemm_blockscaled(a, b, c, sa, sb, transpose_A=dst_transpose, transpose_B=True, clear_accum=True)),
    ]
    return function([gm, *parameters], [*(allocate(t) for t in [l1, scale, a, b, sa, sb, c]), annotation(metadata), *operations]), l1


@pytest.mark.parametrize(
    "dtype,layout,actual_k,tail",
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
    value, l1 = mx_case(layout, actual_k, dtype=dtype)
    result = pad(value, "mx-tail-" + dtype + layout + str(actual_k))
    fills = calls(result, "tl.tileop.fill")
    assert len(fills) == (tail is not None)
    if tail is None:
        return
    (fill,) = fills
    operands, _ = decoded(fill)
    destination, zero = operands
    assert destination.source.same_as(l1) and float(zero.value) == 0
    start, size = tail
    mn = 64 if dtype == "float4_e2m1fn" else 32
    assert range_values(destination) == ([start, 0, size, mn] if layout == "mn-contiguous" else [0, start, mn, size])
    copies = calls(result, "tl.tileop.ascend_copy")
    operations = objects(result, ir.Call)

    def position(call):
        return next(i for i, item in enumerate(operations) if item.same_as(call))

    assert position(copies[0]) < position(fill) < position(copies[1])


def test_mx_padding_rejects_an_incomplete_physical_scale_group():
    value, _ = mx_case("k-contiguous", 96, k_alloc=96)
    with pytest.raises(ValueError, match="divisible by 64"):
        run_pass("AscendInsertOOBPadding", value)


@pytest.mark.parametrize("carrier", ["annotation", "attribute", "kernel"])
def test_padding_reads_native_layout_carriers_without_consuming_them(carrier):
    value = copy_case((10, 8), (16, 32), [0, 0], [16, 32], carrier=carrier)
    result = pad(value, "oob-carrier-" + carrier)
    assert len(calls(result, "tl.tileop.fill")) == 2
    assert "layout_map" in ir.save_json(result)


def test_enclosing_loop_range_proves_copy_is_in_bounds():
    index = ir.Var("i", "int32")
    gm = tensor("gm", [64, 32], "bfloat16", "global")
    l1 = tensor("l1", [16, 32], "bfloat16", "shared.l1")
    body = [
        allocate(l1),
        annotation({"layout_map": {l1: make_ascend_major_k_layout(l1)}}),
        loop(index, 4, [tileop("ascend_copy", subregion(gm, [index * 16, 0], [16, 32]), region(l1))], parallel=False),
    ]
    value = function([gm], body)
    ir.assert_structural_equal(pad(value, "oob-native-loop-context"), value)


def test_adjacent_mx_loads_precede_grouped_dynamic_fills():
    k = ir.Var("k", "int32")
    left, _ = mx_case("k-contiguous", k, suffix="left", parameters=[k])
    right, _ = mx_case("mn-contiguous", k, suffix="right", parameters=[k])
    left_body, right_body = list(left.body.seq), list(right.body.seq)
    body = [*left_body[:-4], *right_body[:-4], left_body[-4], right_body[-4], *left_body[-3:], *right_body[-3:]]
    value = function([left.params[0], right.params[0], k], body)
    result = pad(value, "mx-grouped-dynamic-fills")
    (group,) = [item for item in result.body.seq if isinstance(item, ir.If)]
    assert len(calls(group, "tl.tileop.fill")) == 2
    assert not calls(group, "tl.tileop.ascend_copy")
    gm_positions = [i for i, item in enumerate(result.body.seq) if isinstance(item, ir.Evaluate) and calls(item, "tl.tileop.ascend_copy")][
        :2
    ]
    assert max(gm_positions) < next(i for i, item in enumerate(result.body.seq) if item.same_as(group))
