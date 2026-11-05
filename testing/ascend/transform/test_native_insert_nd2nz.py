"""ND/NZ conversion preserves native tensor views and the existing DMA geometry.

Shape, pitch, and rejection oracles mirror test_ascend_insert_nd2nz.py.  Native
type, lexical metadata, and pointer byte-offset cases additionally check the
new object model.  CCE source checks do not claim CANN or NPU execution.
"""

import math

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape, dtype="bfloat16", scope="shared.dyn", strides=None, origin=0):
    layout = None
    if strides is not None or origin:
        if strides is None:
            strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        layout = tirx.TileLayout(tirx.layout.S[tuple(shape) : tuple(strides)] + origin)
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=layout, align=32))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def declare(value, owner, byte_offset=0):
    pointer = ir.Call("tirx.tensor_data_ptr", [owner])
    if byte_offset:
        pointer = ir.Call("tirx.ptr_byte_offset", [pointer, integer(byte_offset, "int64")], ty=pointer.ty)
    if pointer.ty.element_type != value.ty.dtype:
        pointer = ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType(value.ty.dtype, value.ty.scope))
    return ir.Bind(value, ir.Call("tirx.decl_tensor", [pointer], ty_args=[value.ty]))


def region(value, starts=None, extents=None):
    starts = [0] * len(value.ty.shape) if starts is None else starts
    extents = value.ty.shape if extents is None else extents
    return ir.TensorRegion(
        value, [ir.Range.from_min_extent(start, extent) for start, extent in zip(starts, extents)], ir.TensorRegionType()
    )


def annotated_call(name, operands, metadata=None):
    return tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"), ir.Op.get(name), list(operands), metadata or {}, ir.UnknownLoc()
    )


def copy(src, dst, src_starts=None, dst_starts=None, src_shape=None, dst_shape=None, metadata=None):
    return ir.Evaluate(
        annotated_call("tl.tileop.ascend_copy", [region(src, src_starts, src_shape), region(dst, dst_starts, dst_shape)], metadata)
    )


def attribute(key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def annotation(fields):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def layout(value, kind="compact_nz"):
    return tvm_ffi.get_global_func("tl.make_ascend_" + kind + "_layout")(value)


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def calls(value, name):
    return [call for call in objects(value, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == name]


def operands(call):
    return list(call.args[: int(call.attrs["operand_count"])]) if call.attrs and "operand_count" in call.attrs else list(call.args)


def metadata(call):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"])


def constant(expr):
    return int(tvm_tirx.sym.Analyzer().simplify(expr))


def function(body, buffers=(), definitions=(), layouts=None, params=(), carrier="sblock"):
    statements = [*(allocate(value) for value in buffers), *definitions]
    if carrier == "annotation":
        statements += [annotation({"layout_map": layouts or {}}), *body]
        statements = [attribute("tl.sblock", {"name": "tilelang_root", "no_realize": True}, statements)]
    else:
        statements += [
            attribute("tl.sblock", {"name": "tilelang_root", "no_realize": True, "annotations": {"layout_map": layouts or {}}}, list(body))
        ]
    return tirx.Function(list(params), statements).with_attr("target", tvm_tirx.target.Target("ascend")).with_attr("global_symbol", "main")


def apply(before):
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    transform = tvm_ffi.get_global_func("tl.transform.InsertNd2Nz")()
    after = transform(ir.IRModule({"main": before}))["main"]
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    for call in objects(after, ir.Call):
        call.validate()
    ir.assert_structural_equal(transform(ir.IRModule({"main": after}))["main"], after)
    return after


def scatter_program(*, staged=False, reshape=False, vf=None, padding=1, strides=None, carrier="annotation"):
    rows, cols = 16, 128
    src = tensor("src", [2, rows, cols] if staged else [rows, cols], strides=strides)
    owner = tensor("storage", [rows // 2, cols * 2]) if reshape else src
    dst = tensor("dst", [2, rows + padding, cols] if staged else [rows + padding, cols], "float32")
    i = ir.Var("stage", "int32")
    prefix, shape = ([i], [1]) if staged else ([], [])
    body = [copy(src, dst, prefix + [0, 0], prefix + [0, 0], shape + [rows, cols], shape + [rows, cols])]
    if vf:
        body = [attribute("tl.sblock", {"name": vf, "no_realize": True}, body)]
    if staged:
        body = [ir.For(i, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, body)]
    return function(body, [owner, dst], [declare(src, owner)] if reshape else [], {dst: layout(dst)}, carrier=carrier)


@pytest.mark.parametrize("case", ["plain", "staged", "reshape", "simd", "simt"])
@pytest.mark.parametrize("carrier", ["annotation", "sblock"])
def test_scatter_regions_and_vf_boundary(case, carrier):
    before = scatter_program(
        staged=case == "staged", reshape=case == "reshape", vf={"simd": "SIMD_VF", "simt": "SIMT_VF"}.get(case), carrier=carrier
    )
    after = apply(before)
    if case == "simt":
        ir.assert_structural_equal(after, before)
        return
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    args = operands(scatter)
    assert [constant(x) for x in args[2:5]] == [16, 128, 17]
    assert [str(x.value) for x in args[5:]] == ["bfloat16_t", "float"]
    assert constant(args[0].args[1]) == 16 * 128
    assert constant(args[1].args[1]) == ((128 // 8 - 1) * 17 + 16) * 8
    assert int(metadata(scatter)["nd2nz_logical_regions"]) == 3
    assert not calls(after, "tl.tileop.ascend_copy")
    if case == "staged":
        assert args[0].args[0].indices[0].same_as(args[1].args[0].indices[0])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"padding": 0}, None),
        ({"strides": [256, 1]}, "compact trailing source matrix"),
        ({"staged": True, "strides": [2047, 128, 1]}, "32-byte-aligned source address"),
        ({"staged": True, "strides": [2048, 128, 1]}, None),
    ],
)
def test_scatter_storage_requirements(kwargs, error):
    before = scatter_program(**kwargs)
    if error:
        with pytest.raises(ValueError, match=error):
            apply(before)
    else:
        assert len(calls(apply(before), "tl.ascend_nd2nz_scatter")) == 1


@pytest.mark.parametrize("padding", [0, 1, 7])
@pytest.mark.parametrize("parts", [1, 2, 4, 8])
def test_partial_nz_copy_uses_complete_storage_pitches(padding, parts):
    src, dst = tensor("src", [2, 64 + padding, 128]), tensor("dst", [2, 64, 128], scope="shared.l1")
    rows = 64 // parts
    before = function(
        [copy(src, dst, [1, r, 16], [1, r, 32], [1, rows, 64], [1, rows, 64]) for r in range(0, 64, rows)],
        [src, dst],
        layouts={src: layout(src), dst: layout(dst, "nz")},
    )
    post = calls(apply(before), "tl.ascend_nd2nz_post_copy")
    assert len(post) == parts
    for i, call in enumerate(post):
        args = operands(call)
        assert [constant(x) for x in args[2:6]] == [rows, 64, 64 + padding, 64]
        assert [constant(x) for x in args[1].args[0].indices] == [1, i * rows, 16]
        assert [constant(x) for x in args[0].args[0].indices] == [1, i * rows, 32]
        assert constant(args[1].args[1]) == (3 * (64 + padding) + rows) * 16


@pytest.mark.parametrize("padding", [0, 1, 7])
def test_partial_scatter_uses_destination_storage_pitch(padding):
    src, dst = tensor("src", [16, 128]), tensor("dst", [64 + padding, 256])
    before = function([copy(src, dst, dst_starts=[32, 16], dst_shape=[16, 128])], [src, dst], layouts={dst: layout(dst)})
    (scatter,) = calls(apply(before), "tl.ascend_nd2nz_scatter")
    assert [constant(x) for x in operands(scatter)[2:5]] == [16, 128, 64 + padding]
    assert [constant(x) for x in operands(scatter)[1].args[0].indices] == [32, 16]


@pytest.mark.parametrize(
    "case,error",
    [
        ("out-of-bounds", "provably in bounds"),
        ("column-start", "C0-aligned"),
        ("column-width", "C0-aligned"),
        ("dtype", "dtype conversion"),
        ("destination-layout", "MajorK destination"),
    ],
)
def test_invalid_partial_nz_regions(case, error):
    src, dst = tensor("src", [64, 128]), tensor("dst", [64, 128], "float16" if case == "dtype" else "bfloat16", "shared.l1")
    rows, cols = (65 if case == "out-of-bounds" else 32), (63 if case == "column-width" else 64)
    dst_layout = tvm_ffi.get_global_func("tl.make_ascend_major_mn_layout")(dst, 1) if case == "destination-layout" else layout(dst, "nz")
    before = function(
        [copy(src, dst, [0, 1 if case == "column-start" else 0], src_shape=[rows, cols], dst_shape=[rows, cols])],
        [src, dst],
        layouts={src: layout(src), dst: dst_layout},
    )
    with pytest.raises(ValueError, match=error):
        apply(before)


def access(value, indices, mask):
    return ir.Call("tl.access_ptr", [ir.TensorLoad(value, indices), integer(1), integer(mask)])


def post_copy(src, dst, rows=8, cols=64, src_pitch=64, dst_pitch=64, src_indices=None, dst_indices=None):
    return ir.Evaluate(
        ir.Call(
            "tl.ascend_nd2nz_post_copy",
            [
                access(dst, dst_indices or [0] * len(dst.ty.shape), 2),
                access(src, src_indices or [0] * len(src.ty.shape), 1),
                integer(rows),
                integer(cols),
                integer(src_pitch),
                integer(dst_pitch),
                ir.StringImm("bfloat16_t"),
            ],
            ty="void",
        )
    )


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_explicit_cross_dtype_view_pitch_and_allocation_identity(parts):
    owner, view, dst = tensor("owner", [32, 64], "float32"), tensor("view", [4096]), tensor("dst", [4096], scope="shared.l1")
    rows = 32 // parts
    before = function(
        [post_copy(view, dst, rows, src_indices=[r * 16], dst_indices=[(32 + r) * 16]) for r in range(0, 32, rows)],
        [owner, dst],
        [declare(view, owner)],
    )
    result = apply(before)
    assert len(calls(result, "tl.ascend_nd2nz_post_copy")) == parts
    assert sum(binding.var.same_as(owner) and binding.value.op.name == "tirx.alloc_tensor" for binding in objects(result, ir.Bind)) == 1
    assert [constant(x) for x in owner.ty.shape] == [32, 64]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"src_pitch": 7}, "pitches >= rows"),
        ({"src_pitch": 65544}, "instruction field range"),
        ({"cols": 4096 * 16}, "instruction field range"),
        ({"src_indices": [1]}, "32-byte-aligned"),
        ({"src_indices": [4000]}, "physical span exceeds"),
    ],
)
def test_explicit_post_copy_rejects_invalid_geometry(kwargs, error):
    src, dst = tensor("src", [4096]), tensor("dst", [4096], scope="shared.l1")
    before = function([post_copy(src, dst, **kwargs)], [src, dst])
    with pytest.raises(ValueError, match=error):
        apply(before)


@pytest.mark.parametrize("rows,cols", [(0, 64), (8, 0)])
def test_empty_post_copy_does_not_compute_negative_span(rows, cols):
    src, dst = tensor("src", [4096]), tensor("dst", [4096], scope="shared.l1")
    assert not calls(apply(function([post_copy(src, dst, rows, cols)], [src, dst])), "tl.ascend_nd2nz_post_copy")


@pytest.mark.parametrize("source_rows,rows,pitch", [(1, 1, 1), (8, 8, 8), (17, 8, 32)])
def test_canonical_nz_source_pitch_includes_fractal_rounding(source_rows, rows, pitch):
    src, dst = tensor("src", [source_rows, 64]), tensor("dst", [32, 64], scope="shared.l1")
    before = function(
        [copy(src, dst, src_shape=[rows, 64], dst_shape=[rows, 64])], [src, dst], layouts={src: layout(src, "nz"), dst: layout(dst, "nz")}
    )
    (post,) = calls(apply(before), "tl.ascend_nd2nz_post_copy")
    assert [constant(x) for x in operands(post)[2:6]] == [rows, 64, pitch, 32]


@pytest.mark.parametrize("origin,index,valid", [(16, 0, True), (1024, 0, True), (16, -16, False), (16, 16, False), (-16, 0, False)])
@pytest.mark.parametrize("origin_kind", ["layout", "byte_offset"])
def test_explicit_post_copy_bounds_are_relative_to_view(origin, index, valid, origin_kind):
    owner, dst = tensor("owner", [2048]), tensor("dst", [1024], scope="shared.l1")
    view = tensor("view", [1024], origin=origin if origin_kind == "layout" else 0)
    before = function(
        [post_copy(view, dst, 64, 16, src_indices=[index])],
        [owner, dst],
        [declare(view, owner, origin * 2 if origin_kind == "byte_offset" else 0)],
    )
    if valid:
        assert len(calls(apply(before), "tl.ascend_nd2nz_post_copy")) == 1
    else:
        with pytest.raises(ValueError, match="physical span exceeds"):
            apply(before)


@pytest.mark.parametrize("cols", [16, 64])
def test_dense_ub_to_l1_uses_direct_single_c0_or_bound_scratch(cols):
    src, dst = tensor("src", [16, cols]), tensor("dst", [16, cols], scope="shared.l1")
    before = function([copy(src, dst)], [src, dst], layouts={dst: layout(dst, "nz")}, carrier="annotation")
    after = apply(before)
    posts, scatters = calls(after, "tl.ascend_nd2nz_post_copy"), calls(after, "tl.ascend_nd2nz_scatter")
    assert len(posts) == 1
    assert len(scatters) == (0 if cols == 16 else 1)
    assert constant(operands(posts[0])[4]) == (16 if cols == 16 else 17)
    if cols == 64:
        scratch = operands(scatters[0])[1].args[0].source
        assert operands(posts[0])[1].args[0].source.same_as(scratch)
        assert [constant(x) for x in scratch.ty.shape] == [17 * 64]
        assert sum(binding.var.same_as(scratch) for binding in objects(after, ir.Bind)) == 1


def test_branch_layout_metadata_does_not_leak_to_following_copy():
    src, dst = tensor("src", [16, 128]), tensor("dst", [17, 128], "float32")
    flag = ir.Var("flag", "bool")
    operation = copy(src, dst, dst_shape=[16, 128])
    branch = ir.If(flag, [annotation({"layout_map": {dst: layout(dst)}}), operation], None)
    after = apply(function([branch, operation], [src, dst], params=[flag], carrier="annotation"))
    assert len(calls(after, "tl.ascend_nd2nz_scatter")) == 1
    assert len(calls(after, "tl.tileop.ascend_copy")) == 1


@pytest.mark.parametrize("byte_offset", [1, 2, 31])
def test_post_copy_alignment_checks_bytes_without_truncating_sub_element_offsets(byte_offset):
    owner, view = tensor("owner", [4096]), tensor("view", [2048])
    dst = tensor("dst", [2048], scope="shared.l1")
    before = function([post_copy(view, dst, 16, 16, 16, 16)], [owner, dst], [declare(view, owner, byte_offset)])
    with pytest.raises(ValueError, match="32-byte-aligned"):
        apply(before)


@pytest.mark.parametrize("case", ["scatter", "single_c0", "scratch"])
def test_inserted_intrinsics_reach_real_layout_lowering_and_ascend_codegen(case, tmp_path):
    if case == "scatter":
        before = scatter_program()
    else:
        cols = 16 if case == "single_c0" else 64
        src, dst = tensor("src", [16, cols]), tensor("dst", [16, cols], scope="shared.l1")
        before = function([copy(src, dst)], [src, dst], layouts={dst: layout(dst, "nz")}, carrier="annotation")
    value = apply(before)
    (tmp_path / "inserted.json").write_text(ir.save_json(value))
    for name in ["LayoutInference", "AscendLowerTileOp", "LowerAccessPtr"]:
        with value.attrs["target"]:
            value = tvm_ffi.get_global_func("tl.transform." + name)()(ir.IRModule({"main": value}))["main"]
        assert tirx.analysis.check_well_formed(value), name
        (tmp_path / (name + ".json")).write_text(ir.save_json(value))
    value = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("ascend")(ir.IRModule({"main": value}))["main"]
    module = ir.IRModule({"main": value})
    for transform in [tirx.transform.LowerTensorLayouts, tirx.transform.FlattenTensor, tirx.transform.StmtSimplify]:
        module = transform()(module)
    value = module["main"].with_attr("calling_conv", 2)
    assert tirx.analysis.check_well_formed(value)
    code = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(
        ir.IRModule({"main": value}), value.attrs["target"]
    ).inspect_source("asc")
    (tmp_path / "kernel.asc").write_text(code)
    assert "tl.tileop" not in code
    assert ("ascend_nd2nz_scatter<" in code) == (case != "single_c0")
    if case != "scatter":
        assert "asc_copy_ub2l1" in code
