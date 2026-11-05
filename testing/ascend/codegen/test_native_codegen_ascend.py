"""Native CCE emitter contracts that do not require the CANN SDK."""

import re

import pytest
import tvm_ffi
import tvm_tirx

I = tvm_tirx.ir


def _ptr(name, dtype="float32", scope="global"):
    return I.Var(name, I.PtrType(I.PrimType(dtype), scope))


def _tensor(name, dtype="float32", scope="global", shape=(16,)):
    return I.Var(name, tvm_tirx.tirx.TensorType(shape, dtype, scope=scope, layout="default"))


def _decl(tensor, pointer):
    return I.Bind(tensor, I.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))


def _alloc(tensor):
    return I.Bind(tensor, I.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _call(name, *args, dtype="void"):
    args = [arg if isinstance(arg, I.Expr) else I.StringImm(arg) if isinstance(arg, str) else I.const(arg) for arg in args]
    return I.Call(name, args, ty_args=[I.PrimType(dtype)])


def _block(name, body, **annotations):
    fields = {"name": name, "no_realize": True, "annotations": annotations}
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": I.const(0), "key": "tl.sblock", "value": fields})
    return I.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _thread(var, size, body, tag="threadIdx.x"):
    return I.RegionStmt("tirx.launch_thread", [I.StringImm(tag), I.const(size)], [var], {}, body)


def _source(params, body, tmp_path, *, name="kernel", attrs=None):
    func = tvm_tirx.tirx.Function(params, body, ret_type=I.PrimType("void"))
    func = func.with_attr("global_symbol", name).with_attr("calling_conv", 2)
    for key, value in (attrs or {}).items():
        func = func.with_attr(key, value)
    saved = I.save_json(func)
    module = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(
        tvm_tirx.IRModule({name: func}), tvm_tirx.target.Target("ascend")
    )
    source = module.inspect_source("asc")
    assert I.save_json(func) == saved
    (tmp_path / (name + ".asc")).write_text(source)
    return source


def test_native_ascend_target_and_typed_scalar_memory(tmp_path):
    target = tvm_tirx.target.Target("ascend")
    assert target.launch_grid_only and target.max_vector_bits == 64
    pointer = _ptr("source")
    tensor = _tensor("source_view")
    code = _source([pointer], [_decl(tensor, pointer), I.TensorStore(tensor, [2], I.const(3.25, "float32"))], tmp_path)
    assert "__gm__ float*" in code and "asc_init();" in code
    value = re.search(r"source(?:_view)?\[2\] = float\((0x[0-9a-f.]+p[+-]\d+)f", code)
    assert value and float.fromhex(value.group(1)) == 3.25
    assert "__global__" in code


@pytest.mark.parametrize(
    "scope,qualifier",
    [("shared.dyn", "__ubuf__"), ("shared.l1", "__cbuf__"), ("shared.l0a", "__ca__"), ("shared.l0b", "__cb__"), ("shared.l0c", "__cc__")],
)
def test_native_ascend_memory_spaces(scope, qualifier, tmp_path):
    tensor = _tensor("storage", scope=scope)
    code = _source([], [_alloc(tensor)], tmp_path)
    assert re.search(rf"{qualifier}\s+float\s*\*storage", code)
    assert re.search(rf"\({qualifier}\s+float\s*\*\)0", code)


def test_native_ascend_rejects_static_shared(tmp_path):
    tensor = _tensor("invalid", scope="shared")
    with pytest.raises(RuntimeError, match="Static shared memory"):
        _source([], [_alloc(tensor)], tmp_path)


@pytest.mark.parametrize("allocation_attrs", [None, {}], ids=["omitted", "empty"])
def test_native_ascend_local_allocation_accepts_optional_attrs(allocation_attrs, tmp_path):
    tensor = _tensor("scratch", dtype="int32", scope="local", shape=(8,))
    allocation = I.Bind(tensor, I.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty], attrs=allocation_attrs))
    body = [allocation, I.TensorStore(tensor, [0], I.const(7, "int32"))]
    assert tvm_tirx.tirx.analysis.check_well_formed(tvm_tirx.tirx.Function([], body, ret_type=I.PrimType("void")))
    code = _source([], body, tmp_path)
    assert re.search(r"\bint32_t scratch_ptr\[8\];", code)
    assert "scratch_ptr[0] = 7;" in code


def _unrolled_loop(factor=None):
    var = I.Var("i", "int32")
    annotations = {} if factor is None else {"pragma_unroll_factor": factor}
    return I.For(var, 0, 8, 1, I.ForKind.UNROLLED, [I.Evaluate(var)], annotations=annotations)


@pytest.mark.parametrize("factor", [4, I.const(4, "int32")], ids=["python-integer", "native-integer"])
def test_native_ascend_loop_factor_is_consumed(factor, tmp_path):
    loop = _unrolled_loop(factor)
    assert tvm_tirx.tirx.analysis.check_well_formed(tvm_tirx.tirx.Function([], [loop], ret_type=I.PrimType("void")))
    code = _source([], [loop], tmp_path)
    assert re.findall(r"#pragma unroll[^\n]*", code) == ["#pragma unroll 4"]


def _legacy_unroll_carrier(variable, factor, loop):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": variable, "key": "pragma_unroll_factor", "value": factor})
    return I.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [loop])


@pytest.mark.parametrize("local_factor,expected", [(None, 2), (4, 4)])
def test_native_ascend_loop_factor_precedes_legacy_emitter_hint(local_factor, expected, tmp_path):
    loop = _unrolled_loop(local_factor)
    # Exercise the retained terminal emitter contract directly. The legacy
    # carrier names the following loop's definition, so it is not a general
    # well-formed native producer representation.
    carrier = _legacy_unroll_carrier(loop.var, 2, loop)
    code = _source([], [carrier], tmp_path)
    assert re.findall(r"#pragma unroll[^\n]*", code) == [f"#pragma unroll {expected}"]


def test_native_ascend_unroll_hint_does_not_match_same_name_parameter(tmp_path):
    unrelated = I.Var("i", "int32")
    loop = _unrolled_loop()
    carrier = _legacy_unroll_carrier(unrelated, 2, loop)
    assert tvm_tirx.tirx.analysis.check_well_formed(tvm_tirx.tirx.Function([unrelated], [carrier], ret_type=I.PrimType("void")))
    code = _source([unrelated], [carrier], tmp_path)
    assert re.findall(r"#pragma unroll[^\n]*", code) == ["#pragma unroll"]


def test_native_ascend_simt_helper_captures_and_thread_identity(tmp_path):
    pointer = _ptr("output")
    tensor = _tensor("view")
    tx = I.Var("index", "int32")
    body = [_decl(tensor, pointer), _block("SIMT_VF", [_thread(tx, 16, [I.TensorStore(tensor, [tx], I.const(2, "float32"))])])]
    code = _source([pointer], body, tmp_path)
    assert "__simt_vf__ __launch_bounds__(16) inline void kernel_simt_vf_0" in code
    assert "asc_vf_call<kernel_simt_vf_0>(cce::dim3(16)," in code
    assert "threadIdx.x" in code and "__gm__ float*" in code
    assert code.index("inline void kernel_simt_vf_0") < code.index("void kernel(")


def test_native_ascend_mixed_sections_and_single_vector_subcore(tmp_path):
    body = [
        _block("CUBE", [I.Evaluate(_call("tl.ascend_pipe_barrier", "PIPE_ALL"))]),
        _block(
            "VECTOR",
            [I.Evaluate(_call("tl.ascend_pipe_barrier", "PIPE_V")), I.Evaluate(_call("tl.ascend_pipe_barrier", "PIPE_MTE1"))],
            vector_count=1,
        ),
    ]
    code = _source([], body, tmp_path)
    assert "ASC_IS_AIC" in code and "ASC_IS_AIV" in code
    assert "asc_get_sub_block_id() == 0" in code
    # Preserve the old CCE emitter: PIPE_V is implicit, PIPE_ALL synchronizes
    # all pipelines, and the remaining pipes use the explicit pipe intrinsic.
    assert code.count("asc_sync();") == 1
    assert "asc_sync_pipe(PIPE_MTE1);" in code and "PIPE_V" not in code


def test_native_ascend_source_helper_names_do_not_collide(tmp_path):
    tx = I.Var("tx", "int32")
    block = _block("SIMT_VF", [_thread(tx, 4, [I.Evaluate(tx)])], **{"tl.vf_source_index": I.const(9)})
    code = _source([], [block, block], tmp_path)
    names = re.findall(r"inline void (kernel_simt_vf_\w+)\(", code)
    assert len(names) == 2 and len(set(names)) == 2
    assert all("_9" in name for name in names)


def test_native_ascend_simd_vector_ops_and_generic_vector_rejection(tmp_path):
    mask = I.Var("mask", "boolx256")
    vec = I.Var("value", "float32x64")
    body = [
        I.Bind(mask, _call("tl.simd.pset", 32, "PAT_ALL", dtype="boolx256")),
        I.Bind(vec, _call("tl.simd.vdup", I.const(1, "float32"), mask, "MODE_ZEROING", dtype="float32x64")),
    ]
    code = _source([], [_block("SIMD_VF", body)], tmp_path)
    assert "__simd_vf__ inline void" in code and "vector_bool mask" in code
    assert "asc_create_mask_b32(PAT_ALL)" in code and "vdup" in code
    invalid = I.Evaluate(I.prim.Broadcast(I.const(1, "float32"), 64))
    with pytest.raises(RuntimeError, match="Generic vector Broadcast"):
        _source([], [_block("SIMD_VF", [invalid])], tmp_path, name="invalid")


def test_native_ascend_typed_dma_and_padding_width(tmp_path):
    gm = _ptr("gm", "float16", "global")
    ub = _ptr("ub", "float16", "shared.dyn")
    copy = _call("tl.ascend_copy_gm_to_ubuf", ub, gm, 0, 1, 32, 0, 2, 0, 0, 0, 0)
    code = _source([gm, ub], [I.Evaluate(copy)], tmp_path)
    assert "asc_copy_gm2ub_align" in code
    assert "__ubuf__ uint16_t" in code and "__gm__ uint16_t" in code


def test_native_ascend_typed_pointer_reinterpret_and_byte_offset(tmp_path):
    source = _ptr("source", "uint8")
    shifted = _ptr("shifted", "float32")
    data = I.Call("tirx.reinterpret", [source], ty=shifted.ty)
    data = tvm_tirx.tirx.op.ptr_byte_offset(data, I.const(4), ty=shifted.ty)
    tensor = _tensor("view")
    code = _source([source], [I.Bind(shifted, data), _decl(tensor, shifted), I.TensorStore(tensor, [0], I.const(1, "float32"))], tmp_path)
    assert "__gm__ float" in code and "+ 4" in code
    assert "__gm__ char" in code


def test_native_ascend_vf_rejects_nested_sections(tmp_path):
    with pytest.raises(RuntimeError, match="Nested Ascend VF"):
        _source([], [_block("SIMD_VF", [_block("SIMD_VF", [])])], tmp_path)


@pytest.mark.parametrize("source_dtype", ["uint8", "int32", "float64"])
def test_native_ascend_direct_pointer_reinterpret_bind(tmp_path, source_dtype):
    source = _ptr("source", source_dtype)
    shifted = _ptr("shifted", "float32")
    data = tvm_tirx.tirx.op.reinterpret(source, ty=shifted.ty)
    tensor = _tensor("view")
    body = [I.Bind(shifted, data), _decl(tensor, shifted), I.TensorStore(tensor, [0], I.const(1, "float32"))]
    assert tvm_tirx.tirx.analysis.check_well_formed(tvm_tirx.tirx.Function([source], body, ret_type=I.PrimType("void")))
    code = _source([source], body, tmp_path)
    assert "__gm__ float *shifted" in code
    assert "source" in code and "shifted[0]" in code


@pytest.mark.parametrize("source_dtype", ["int64", "uint64"])
def test_native_ascend_integer_reinterpret_bind_stays_scoped(tmp_path, source_dtype):
    source = I.Var("address", I.PrimType(source_dtype))
    shifted = _ptr("shifted", "float32")
    data = tvm_tirx.tirx.op.reinterpret(source, ty=shifted.ty)
    tensor = _tensor("view")
    body = [I.Bind(shifted, data), _decl(tensor, shifted), I.TensorStore(tensor, [0], I.const(1, "float32"))]
    assert tvm_tirx.tirx.analysis.check_well_formed(tvm_tirx.tirx.Function([source], body, ret_type=I.PrimType("void")))
    code = _source([source], body, tmp_path)
    assert "__gm__ float *shifted" in code
    assert "address" in code and "shifted[0]" in code


def test_subcore_binders_are_read_once_and_keep_distinct_identity(tmp_path):
    output = _tensor("output", "int32")
    pointer = _ptr("pointer", "int32")
    body = [_decl(output, pointer)]
    for offset in (0, 2):
        sid = I.Var("sid", "int32")
        body.append(_thread(sid, 2, [I.TensorStore(output, [offset], sid), I.TensorStore(output, [offset + 1], sid)], tag="cthread"))
    source = _source([pointer], body, tmp_path)
    definitions = re.findall(r"int32_t (sid\w*) = asc_get_sub_block_id\(\);", source)
    assert len(definitions) == len(set(definitions)) == 2
    assert source.count("asc_get_sub_block_id()") == 2
    for offset, name in zip((0, 2), definitions):
        assert f"pointer[{offset}] = {name};" in source
        assert f"pointer[{offset + 1}] = {name};" in source
