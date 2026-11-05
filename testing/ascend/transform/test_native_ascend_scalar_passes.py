"""Native Ascend merging, explicit unroll and physical GM dcache contracts."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, dtype="float32", shape=(64,), scope="global"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope))


def pointer(name, dtype="float32", scope="global"):
    return ir.Var(name, ir.PtrType(ir.PrimType(dtype), scope))


def call(name, *args, dtype="void", metadata=None):
    attrs = None
    args = list(args)
    if metadata is not None:
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(metadata)
        attrs = {"schema_version": 1, "operand_count": len(args), "annotation_schema": schema, "test_static": 17}
        args.append(payload)
    result = ir.Call(name, args, ty_args=[ir.PrimType(dtype)], attrs=attrs)
    result.validate()
    return result


def attribute(key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": integer(0), "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def vf(body, kind="SIMD_VF", form="named"):
    tag = "tl.simdvf_scope" if kind == "SIMD_VF" else "tl.simtvf_scope"
    if form == "direct":
        return attribute(tag, 1, body)
    annotations = {tag: 1} if form == "annotation" else {}
    return attribute(
        "tl.sblock", {"name": "ordinary" if form == "annotation" else kind, "no_realize": True, "annotations": annotations}, body
    )


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def declare(value, data):
    return ir.Bind(value, ir.Call("tirx.decl_tensor", [data], ty_args=[value.ty]))


def access(value, index=0, mask=3):
    return ir.Call(
        "tl.access_ptr", [ir.TensorLoad(value, [integer(index) if isinstance(index, int) else index]), integer(1), integer(mask)]
    )


def objects(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


def calls(value, name):
    return [item for item in objects(value, ir.Call) if isinstance(item.op, ir.Op) and item.op.name == name]


def apply(name, params, body, tmp_path):
    before = tirx.Function(params, body).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(before)
    original = ir.save_json(before)
    operation = tvm_ffi.get_global_func("tl.transform." + name)()
    result = operation(tvm.IRModule({"main": before}))
    assert tirx.analysis.check_well_formed(result)
    for expr in objects(result, ir.Call):
        expr.validate()
    assert ir.save_json(before) == original
    ir.assert_structural_equal(operation(result), result)
    (tmp_path / "before.json").write_text(original)
    (tmp_path / "after.json").write_text(ir.save_json(result["main"]))
    return result["main"]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int8", "uint8", "int16", "uint16", "int32", "uint32"])
@pytest.mark.parametrize("merging", [False, True])
def test_simd_destination_is_read_write_and_result_becomes_void(dtype, merging, tmp_path):
    bits = ir.PrimType(dtype).dtype.bits
    vector_dtype = f"{dtype}x{2048 // bits}"
    dst = tensor("dst", vector_dtype, (1,), "local")
    mask = ir.Var("mask", "boolx256")
    mode = "MODE_MERGING" if merging else "MODE_ZEROING"
    value = call("tl.simd.vadds", ir.TensorLoad(dst, [0]), ir.const(1, dtype), mask, ir.StringImm(mode), dtype=vector_dtype)
    body = [ir.TensorStore(dst, [0], value)]
    result = apply("LegalizeSimdMerging", [dst, mask], body, tmp_path)
    (output,) = calls(result, "tl.simd.vadds")
    if not merging:
        ir.assert_structural_equal(result.body, ir.SeqStmt(body))
        return
    assert output.ty == ir.PrimType("void") and not output.ty_args
    destination, source = output.args[:2]
    assert destination.op.name == "tl.access_ptr"
    assert destination.args[1].value == 1 and destination.args[2].value == 3
    ir.assert_structural_equal(destination.args[0], source)
    # Inactive-lane oracle is independent of the spelling of the new IR.
    old = np.arange(2048 // bits, dtype=np.int64) % 29
    enabled = np.arange(len(old)) % 3 == 0
    expected = np.where(enabled, old + 1, old)
    actual = old.copy() if int(destination.args[2].value) & 1 else np.zeros_like(old)
    actual[enabled] = (old + 1)[enabled]
    np.testing.assert_array_equal(actual, expected)


def test_merging_preserves_metadata_and_conditional_store(tmp_path):
    dst = tensor("dst", "float32x64", (1,), "local")
    source, mask, meta = ir.Var("source", "float32x64"), ir.Var("mask", "boolx256"), ir.Var("meta", "int32")
    cond = ir.Var("cond", "bool")
    value = call(
        "tl.simd.vadds", source, ir.const(1, "float32"), mask, ir.StringImm("MODE_MERGING"), dtype="float32x64", metadata={"trace": meta}
    )
    result = apply("LegalizeSimdMerging", [dst, source, mask, meta, cond], [ir.If(cond, [ir.TensorStore(dst, [0], value)], None)], tmp_path)
    (merged,) = calls(result, "tl.simd.vadds")
    assert int(merged.attrs["operand_count"]) == 5 and int(merged.attrs["test_static"]) == 17
    decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(merged.args[-1], merged.attrs["annotation_schema"])
    assert decoded["trace"].same_as(meta)
    assert len(objects(result, ir.If)) == 1


@pytest.mark.parametrize("name,dtype,inputs", [("vxor", "int32x64", 2), ("vnot", "int32x64", 1), ("vcpadd", "float32x64", 1)])
def test_merging_fixed_intrinsics_accept_void_destination(name, dtype, inputs, tmp_path):
    dst, source, mask = tensor("dst", dtype, (1,), "local"), ir.Var("source", dtype), ir.Var("mask", "boolx256")
    operands = [source] * inputs + [mask, ir.StringImm("MODE_MERGING")]
    value = call("tl.simd." + name, *operands, dtype=dtype)
    result = apply("LegalizeSimdMerging", [dst, source, mask], [ir.TensorStore(dst, [0], value)], tmp_path)
    (merged,) = calls(result, "tl.simd." + name)
    assert merged.ty == ir.PrimType("void") and len(merged.args) == len(operands) + 1


@pytest.mark.parametrize("kind", ["bind", "evaluate", "nested"])
def test_merging_without_mutable_destination_is_rejected(kind, tmp_path):
    source, mask = ir.Var("source", "float32x64"), ir.Var("mask", "boolx256")
    merged = call("tl.simd.vadds", source, ir.const(1, "float32"), mask, ir.StringImm("MODE_MERGING"), dtype="float32x64")
    value = ir.Var("value", "float32x64")
    body = [ir.Bind(value, merged)] if kind == "bind" else [ir.Evaluate(merged)]
    if kind == "nested":
        dst = tensor("dst", "float32x64", (1,), "local")
        body = [allocate(dst), ir.TensorStore(dst, [0], ir.prim.Add(merged, source))]
    with pytest.raises(RuntimeError, match="read-modify-write"):
        apply("LegalizeSimdMerging", [source, mask], body, tmp_path)


def loop(var, begin, end, body, step=1, explicit=True, kind=ir.ForKind.UNROLLED):
    attrs = {} if explicit is None else {"pragma_unroll_explicit": explicit}
    return ir.For(var, integer(begin, str(var.ty)), integer(end, str(var.ty)), integer(step, str(var.ty)), kind, body, attrs)


@pytest.mark.parametrize(
    "begin,end,step,dtype", [(0, 4, 1, "int32"), (2, 9, 3, "int32"), (-100, 100, 1, "int8"), (9, 2, 1, "uint32"), (3, 3, 1, "int32")]
)
def test_explicit_unroll_preserves_native_iteration_values(begin, end, step, dtype, tmp_path):
    i = ir.Var("i", dtype)
    trace = ir.Evaluate(tirx.call_extern("record_trace", i, ty="void"))
    result = apply("UnrollLoopSkipVF", [], [loop(i, begin, end, [trace], step)], tmp_path)
    assert not objects(result, ir.For)
    assert [int(stmt.value.args[1].value) for stmt in result.body.seq] == list(range(begin, end, step))


@pytest.mark.parametrize(
    "explicit,kind", [(False, ir.ForKind.UNROLLED), (None, ir.ForKind.UNROLLED), (True, ir.ForKind.DEFAULT), (True, ir.ForKind.PARALLEL)]
)
def test_only_explicit_unrolled_loops_expand(explicit, kind, tmp_path):
    i = ir.Var("i", "int32")
    result = apply("UnrollLoopSkipVF", [], [loop(i, 0, 3, [ir.Evaluate(i)], explicit=explicit, kind=kind)], tmp_path)
    assert len(objects(result, ir.For)) == 1


@pytest.mark.parametrize("kind", ["SIMD_VF", "SIMT_VF"])
@pytest.mark.parametrize("form", ["named", "direct", "annotation"])
def test_vf_scopes_preserve_nested_unrolls(kind, form, tmp_path):
    i = ir.Var("i", "int32")
    protected = vf([attribute("ordinary", "SIMD_VF", [loop(i, 0, 3, [ir.Evaluate(i)])])], kind, form)
    result = apply("UnrollLoopSkipVF", [], [protected], tmp_path)
    ir.assert_structural_equal(result.body.seq[0], protected)


def test_ordinary_attribute_value_is_not_a_vf_scope(tmp_path):
    i = ir.Var("i", "int32")
    body = [attribute("ordinary", "SIMD_VF", [loop(i, 0, 3, [ir.Evaluate(i)])])]
    result = apply("UnrollLoopSkipVF", [], body, tmp_path)
    assert not objects(result, ir.For)


def test_unroll_renews_local_type_and_vf_capture_identities(tmp_path):
    i = ir.Var("same_name", "int32")
    external = ir.Var("same_name", "int32")
    local = tensor("allocation", "float32", (i + 1,), "local")
    value = ir.Var("value", "int32")
    body = [allocate(local), ir.Bind(value, i + external), vf([ir.TensorStore(local, [i], ir.const(1, "float32")), ir.Evaluate(value)])]
    result = apply("UnrollLoopSkipVF", [external], [loop(i, 0, 3, body)], tmp_path)
    definitions = [bind.var for bind in objects(result, ir.Bind)]
    assert len(definitions) == 6 and len({hash(var) for var in definitions}) == 6
    assert not any(var.same_as(i) for var in objects(result, ir.Var))
    local_shapes = [str(var.ty.shape[0]) for var in definitions if isinstance(var.ty, tirx.TensorType)]
    assert len(local_shapes) == 3
    assert any(var.same_as(external) for var in objects(result, ir.Var))


@pytest.mark.parametrize("control", [ir.Break, ir.Continue])
def test_unroll_rejects_control_that_targets_expanded_loop(control, tmp_path):
    i = ir.Var("i", "int32")
    with pytest.raises(ValueError, match="break or continue"):
        apply("UnrollLoopSkipVF", [], [loop(i, 0, 3, [control()])], tmp_path)


def test_unroll_allows_control_targeting_preserved_inner_loop(tmp_path):
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    inner = loop(j, 0, 2, [ir.Break()], explicit=False, kind=ir.ForKind.DEFAULT)
    result = apply("UnrollLoopSkipVF", [], [loop(i, 0, 3, [inner])], tmp_path)
    assert len(objects(result, ir.For)) == 3 and len(objects(result, ir.Break)) == 3


@pytest.mark.parametrize("begin,end,step,dtype", [(126, 127, 2, "int8"), (0, 3, 0, "int32"), (0, 3, -1, "int32")])
def test_unroll_rejects_non_finite_or_wrapping_range(begin, end, step, dtype, tmp_path):
    i = ir.Var("i", dtype)
    with pytest.raises(ValueError, match="constant loop extent"):
        apply("UnrollLoopSkipVF", [], [loop(i, begin, end, [ir.Evaluate(i)], step)], tmp_path)


@pytest.mark.parametrize("writer", ["scalar", "mte3", "fixpipe"])
@pytest.mark.parametrize("alias", ["direct", "tensor_data", "byte_offset", "reinterpret"])
def test_dcache_tracks_writable_storage_and_keeps_readonly_gm(writer, alias, tmp_path):
    writable, readonly = tensor("writable"), tensor("readonly")
    local = tensor("local", scope="shared.dyn")
    view = writable
    prelude = [allocate(local)]
    if alias != "direct":
        dtype = "int32" if alias == "reinterpret" else "float32"
        view = tensor("view", dtype)
        data = writable.data
        if alias == "byte_offset":
            data = ir.Call("tirx.ptr_byte_offset", [data, integer(4)])
        if alias == "reinterpret":
            data = ir.Call("tirx.reinterpret", [data], ty=ir.PtrType(ir.PrimType(dtype), "global"))
        prelude.append(declare(view, data))
    read = ir.TensorStore(local, [0], ir.TensorLoad(writable, [0]) + ir.TensorLoad(readonly, [0]))
    if writer == "scalar":
        write = ir.TensorStore(view, [0], ir.const(2, str(view.ty.dtype)))
    else:
        name = "tl.ascend_copy_ubuf_to_gm" if writer == "mte3" else "tl.ascend_copy_matrix_cc_to_gm"
        argc = 8 if writer == "mte3" else 25
        write = ir.Evaluate(call(name, access(view, mask=2), access(local, mask=1), *[integer(0)] * (argc - 2)))
    result = apply("MarkScalarDcacheBypass", [writable, readonly], [*prelude, read, write], tmp_path)
    (bypass,) = calls(result, "tl.ascend_read_gm_bypass_dcache")
    assert bypass.args[0].args[0].source.same_as(writable)
    assert len(calls(result, "tl.ascend_write_gm_bypass_dcache")) == (writer == "scalar")
    assert any(load.source.same_as(readonly) for load in objects(result, ir.TensorLoad))


@pytest.mark.parametrize("kind", ["SIMD_VF", "SIMT_VF"])
@pytest.mark.parametrize("form", ["named", "direct", "annotation"])
def test_vf_dcache_scope_restores_following_scalar_siblings(kind, form, tmp_path):
    data = tensor("data")
    store = ir.TensorStore(data, [0], ir.TensorLoad(data, [0]) + ir.const(1, "float32"))
    protected = vf([vf([store], kind, form), store], kind, form)
    result = apply("MarkScalarDcacheBypass", [data], [protected, store], tmp_path)
    ir.assert_structural_equal(result.body.seq[0], protected)
    assert len(calls(result, "tl.ascend_read_gm_bypass_dcache")) == 1
    assert len(calls(result, "tl.ascend_write_gm_bypass_dcache")) == 1


@pytest.mark.parametrize("form", ["address_of", "access_ptr"])
def test_address_markers_are_not_scalar_reads_but_indices_are(form, tmp_path):
    data, indices = tensor("data"), tensor("indices", "int32")
    load = ir.TensorLoad(data, [ir.TensorLoad(indices, [0])])
    address = ir.Call("tirx.address_of", [load]) if form == "address_of" else ir.Call("tl.access_ptr", [load, integer(1), integer(3)])
    body = [ir.Evaluate(address), ir.TensorStore(data, [0], ir.const(1, "float32")), ir.TensorStore(indices, [0], integer(0))]
    result = apply("MarkScalarDcacheBypass", [data, indices], body, tmp_path)
    (read,) = calls(result, "tl.ascend_read_gm_bypass_dcache")
    assert read.args[0].args[0].source.same_as(indices)
    assert isinstance(result.body.seq[0].value.args[0], ir.TensorLoad)


def test_unknown_pointer_write_is_conservative(tmp_path):
    data = tensor("data")
    unknown = tirx.call_extern("opaque_global_storage", ty=ir.PtrType(ir.PrimType("float32"), "global"))
    write = ir.Evaluate(call("tl.ascend_copy_ubuf_to_gm", unknown, data.data, *[integer(0)] * 6))
    result = apply("MarkScalarDcacheBypass", [data], [ir.Evaluate(ir.TensorLoad(data, [0])), write], tmp_path)
    assert len(calls(result, "tl.ascend_read_gm_bypass_dcache")) == 1


@pytest.mark.parametrize("shape,dtype,indices", [((4, 4), "float32", [0, 0]), ((4,), "float32x4", [0])])
def test_dcache_keeps_multidimensional_and_vector_accesses(shape, dtype, indices, tmp_path):
    data = tensor("data", dtype, shape)
    source = ir.Var("source", dtype)
    result = apply("MarkScalarDcacheBypass", [data, source], [ir.TensorStore(data, indices, source)], tmp_path)
    assert not calls(result, "tl.ascend_write_gm_bypass_dcache")


def test_native_cce_source_contains_merging_and_scalar_bypass(tmp_path):
    data = tensor("data")
    vec = tensor("dst", "float32x64", (1,), "local")
    mask = ir.Var("mask", "boolx256")
    value = ir.Var("value", "float32x64")
    protected = vf(
        [
            allocate(vec),
            ir.Bind(mask, call("tl.simd.pset", integer(32), ir.StringImm("PAT_VL8"), dtype="boolx256")),
            ir.Bind(value, call("tl.simd.vdup", ir.const(1, "float32"), mask, ir.StringImm("MODE_ZEROING"), dtype="float32x64")),
            ir.TensorStore(vec, [0], value),
            ir.TensorStore(
                vec, [0], call("tl.simd.vadds", value, ir.const(1, "float32"), mask, ir.StringImm("MODE_MERGING"), dtype="float32x64")
            ),
        ]
    )
    before = apply(
        "LegalizeSimdMerging", [data], [protected, ir.TensorStore(data, [0], ir.TensorLoad(data, [0]) + ir.const(1, "float32"))], tmp_path
    )
    result = apply("MarkScalarDcacheBypass", before.params, before.body, tmp_path).with_attr("calling_conv", 2)
    # Native source codegen expects pointer parameters after kernel ABI lowering.
    parameter = pointer("parameter")
    result = (
        tirx.Function([parameter], [declare(data, parameter), *result.body.seq])
        .with_attr("global_symbol", "native_ascend")
        .with_attr("calling_conv", 2)
    )
    result = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(tvm.IRModule({"native_ascend": result}))["native_ascend"]
    module = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(
        tvm.IRModule({"native_ascend": result}), tvm.target.Target("ascend")
    )
    source = module.inspect_source("asc")
    (tmp_path / "native_ascend.asc").write_text(source)
    assert "__simd_vf__" in source and "vadds" in source
    assert "tl::read_gm_bypass_dcache" in source and "tl::write_gm_bypass_dcache" in source
