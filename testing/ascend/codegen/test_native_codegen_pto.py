"""PTO source contracts over native IR; source validation needs no NPU SDK."""

import ast
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


def _source(params, body, tmp_path, *, name="kernel"):
    func = tvm_tirx.tirx.Function(params, body, ret_type=I.PrimType("void"))
    func = func.with_attr("global_symbol", name).with_attr("calling_conv", 2)
    module = tvm_ffi.get_global_func("target.build.tilelang_pto_without_compile")(
        tvm_tirx.IRModule({name: func}), tvm_tirx.target.Target({"kind": "ascend", "keys": ["pto"]})
    )
    source = module.inspect_source("py")
    (tmp_path / (name + ".py")).write_text(source)
    ast.parse(source)
    return source


def _data(tensor):
    return tvm_tirx.tirx.op.tensor_data_ptr(tensor)


def _offset(pointer, byte_offset):
    return tvm_tirx.tirx.op.ptr_byte_offset(pointer, I.const(byte_offset), ty=pointer.ty)


def _mad(a, b, c, flag=0):
    return I.Evaluate(_call("tl.ascend_mad", _data(c), _data(a), _data(b), 16, 16, 16, 0, 0, 0, flag))


def _matrix_body(body_builder):
    a = _tensor("a", "float32", "shared.l0a", (256,))
    b = _tensor("b", "float32", "shared.l0b", (256,))
    c = _tensor("c", "float32", "shared.l0c", (256,))
    return [_alloc(a), _alloc(b), _alloc(c), *body_builder(a, b, c)]


def test_native_pto_typed_scalar_memory(tmp_path):
    pointer = _ptr("source")
    tensor = _tensor("view")
    code = _source([pointer], [_decl(tensor, pointer), I.TensorStore(tensor, [2], I.const(3.25, "float32"))], tmp_path)
    assert 'source: pto.ptr(pto.f32, "gm")' in code
    literal = re.search(r"float.fromhex\('([^']+)'\)", code)
    assert literal and float.fromhex(literal.group(1)) == 3.25
    assert "pto.store(" in code and ", view, 2)" in code and "pto.init_core()" in code


def test_native_pto_tensor_parameter_and_native_data_pointer(tmp_path):
    tensor = _tensor("a")
    pointer = _ptr("data")
    code = _source([tensor], [I.Bind(pointer, _data(tensor)), I.TensorStore(tensor, [3], I.const(1, "float32"))], tmp_path)
    assert "data = a" in code and ", a, 3)" in code
    assert "buffer_map" not in code


def test_native_pto_pointer_byte_offset_keeps_units_and_dtype(tmp_path):
    raw = _ptr("raw", "uint8")
    data = _ptr("data", "float32")
    cast = I.Call("tirx.reinterpret", [raw], ty=data.ty)
    shifted = _offset(cast, 4)
    tensor = _tensor("view")
    code = _source([raw], [I.Bind(data, shifted), _decl(tensor, data), I.TensorStore(tensor, [0], I.const(1, "float32"))], tmp_path)
    assert "pto.addptr(" in code and "pto.castptr(" in code and ", 4)" in code
    assert 'pto.ptr(pto.f32, "gm")' in code
    assert ", view, 0)" in code


@pytest.mark.parametrize("offset", [0, 32])
def test_native_pto_typed_dma_and_offsets(offset, tmp_path):
    gm = _ptr("gm", "float16")
    ub = _ptr("ub", "float16", "shared.dyn")
    copy = _call("tl.ascend_copy_gm_to_ubuf", _offset(ub, offset), gm, 0, 1, 32, 0, 0, 0, 0, 0, 0)
    code = _source([gm, ub], [I.Evaluate(copy)], tmp_path)
    assert "pto.mte_gm_ub(" in code
    if offset:
        assert ", 32)" in code


def test_native_pto_dma_rejects_unaligned_ub_offset(tmp_path):
    gm = _ptr("gm", "float16")
    ub = _ptr("ub", "float16", "shared.dyn")
    copy = _call("tl.ascend_copy_ubuf_to_gm", gm, _offset(ub, 2), 0, 1, 32, 0, 0, 0)
    with pytest.raises((ValueError, RuntimeError), match="32.*align|align.*32"):
        _source([gm, ub], [I.Evaluate(copy)], tmp_path)


def test_native_pto_simt_lexical_thread_and_shared_sync(tmp_path):
    pointer = _ptr("output")
    tensor = _tensor("view")
    thread = I.Var("thread", "int32")
    sync = I.Evaluate(I.Call("tirx.gpu_storage_sync", [I.StringImm("shared")]))
    body = [
        _decl(tensor, pointer),
        _block("SIMT_VF", [_thread(thread, 16, [I.TensorStore(tensor, [thread], I.const(2, "float32")), sync])]),
    ]
    code = _source([pointer], body, tmp_path)
    assert "with pto.simt(16, 1, 1):" in code and "pto.get_tid_x()" in code
    assert "pto.syncthreads()" in code


def test_native_pto_mixed_sections_and_single_vector_subcore(tmp_path):
    body = _matrix_body(
        lambda a, b, c: [
            _block("CUBE", [_mad(a, b, c)]),
            _block("VECTOR", [I.Evaluate(_call("tl.ascend_pipe_barrier", "PIPE_V"))], vector_count=1),
        ]
    )
    code = _source([], body, tmp_path)
    assert "tl.finalize_mixed_kernel" in code and 'tl.mixed_kernel_section("cube")' in code and 'tl.mixed_kernel_section("vector")' in code
    assert "pto.get_subblock_idx()" in code


def test_native_pto_direction_nested_guards(tmp_path):
    flag = I.Var("flag", "int32")
    set_n = I.Evaluate(_call("tl.ascend_set_mmad_direction", "n"))
    set_m = I.Evaluate(_call("tl.ascend_set_mmad_direction", "m"))
    body = _matrix_body(
        lambda a, b, c: [
            I.If(I.prim.And(flag > 0, flag < 10), [set_n], None),
            I.If(flag > 0, [I.If(flag < 10, [_mad(a, b, c), set_m], None)], None),
            _mad(a, b, c),
        ]
    )
    code = _source([flag], body, tmp_path)
    directions = ["n_dir=True" in line for line in code.splitlines() if line.strip().startswith(("pto.mad(", "pto.mad_acc("))]
    assert directions == [True, False]


def test_native_pto_direction_ambiguity_is_rejected(tmp_path):
    flag = I.Var("flag", "int32")
    body = _matrix_body(lambda a, b, c: [I.If(flag > 0, [I.Evaluate(_call("tl.ascend_set_mmad_direction", "n"))], None), _mad(a, b, c)])
    with pytest.raises((ValueError, RuntimeError), match="direction depends on control flow"):
        _source([flag], body, tmp_path)


@pytest.mark.parametrize("kind", ["then", "else", "while", "for"])
def test_native_pto_setter_only_scopes_are_valid_python(kind, tmp_path):
    flag = I.Var("flag", "int32")
    setter = I.Evaluate(_call("tl.ascend_set_mmad_direction", "m"))
    if kind == "then":
        stmt = I.If(flag > 0, [setter], None)
    elif kind == "else":
        stmt = I.If(flag > 0, [I.Evaluate(_call("tl.ascend_pipe_barrier", "PIPE_ALL"))], [setter])
    elif kind == "while":
        stmt = I.While(flag > 0, [setter])
    else:
        stmt = I.For(I.Var("i", "int32"), 0, 1, 1, I.ForKind.UNROLLED, [setter], {"pragma_unroll_explicit": True})
    _source([flag], [stmt], tmp_path)


def test_native_pto_shift_rejects_dynamic_and_excessive_amount(tmp_path):
    value, amount = I.Var("value", "int32"), I.Var("amount", "int32")
    for expression, message in [(I.prim.LShift(value, amount), "constant non-negative"), (I.prim.RShift(value, I.const(32)), "bit width")]:
        with pytest.raises(ValueError, match=message):
            _source([value, amount], [I.Evaluate(expression)], tmp_path)


def test_native_pto_shared_static_is_rejected(tmp_path):
    tensor = _tensor("storage", scope="shared")
    with pytest.raises(RuntimeError, match="Static shared memory"):
        _source([], [_alloc(tensor)], tmp_path)


def _directions(code):
    return ["n_dir=True" in line for line in code.splitlines() if line.strip().startswith(("pto.mad(", "pto.mad_acc("))]


def test_native_pto_mutable_guards_are_not_correlated(tmp_path):
    flag = I.Var("flag", "int32")
    value = _tensor("value", "int32", "local.var", (1,))

    def body(a, b, c):
        return [
            _alloc(value),
            I.TensorStore(value, [0], flag),
            I.If(I.TensorLoad(value, [0]) > 0, [I.Evaluate(_call("tl.ascend_set_mmad_direction", "n"))], None),
            I.TensorStore(value, [0], I.const(1)),
            I.If(I.TensorLoad(value, [0]) > 0, [_mad(a, b, c)], None),
        ]

    with pytest.raises(RuntimeError, match="direction depends on control flow"):
        _source([flag], _matrix_body(body), tmp_path)


@pytest.mark.parametrize("begin,end,step", [(0, 0, 1), (0, 1, 1), (3, 2, -1)])
def test_native_pto_loop_direction_zero_and_one_iteration(begin, end, step, tmp_path):
    def body(a, b, c):
        loop = I.For(
            I.Var("i", "int32"),
            begin,
            end,
            step,
            I.ForKind.DEFAULT,
            [_mad(a, b, c), I.Evaluate(_call("tl.ascend_set_mmad_direction", "n"))],
        )
        return [loop, _mad(a, b, c)]

    code = _source([], _matrix_body(body), tmp_path)
    assert "range(" in code
    assert _directions(code)[-1] is (begin != end)


def test_native_pto_loop_backedge_direction_conflict(tmp_path):
    def body(a, b, c):
        return [
            I.For(I.Var("i", "int32"), 0, 2, 1, I.ForKind.DEFAULT, [_mad(a, b, c), I.Evaluate(_call("tl.ascend_set_mmad_direction", "n"))])
        ]

    with pytest.raises(RuntimeError, match="direction depends on control flow"):
        _source([], _matrix_body(body), tmp_path)


@pytest.mark.parametrize("break_kind", ["native", "tilelang"])
def test_native_pto_break_discards_unreachable_direction(break_kind, tmp_path):
    def body(a, b, c):
        stop = I.Break() if break_kind == "native" else I.Evaluate(_call("tl.loop_break"))
        loop = I.While(
            I.const(True),
            [I.Evaluate(_call("tl.ascend_set_mmad_direction", "n")), stop, I.Evaluate(_call("tl.ascend_set_mmad_direction", "m"))],
        )
        return [loop, _mad(a, b, c)]

    code = _source([], _matrix_body(body), tmp_path)
    assert _directions(code) == [True]


@pytest.mark.parametrize("mode,spelling", [(0, None), (1, "ROUND_AWAY"), (2, "ROUND_EVEN")])
def test_native_pto_hf32_mode_reaches_fp32_mad(mode, spelling, tmp_path):
    body = _matrix_body(lambda a, b, c: [I.Evaluate(_call("tl.ascend_set_hf32_mode", mode)), _mad(a, b, c)])
    code = _source([], body, tmp_path)
    if spelling is None:
        assert "tf32_mode=" not in code
    else:
        assert "tf32_mode=pto.Tf32Mode." + spelling in code


def test_native_pto_hf32_ambiguous_mode_is_rejected(tmp_path):
    flag = I.Var("flag", "int32")
    body = _matrix_body(lambda a, b, c: [I.If(flag > 0, [I.Evaluate(_call("tl.ascend_set_hf32_mode", 2))], None), _mad(a, b, c)])
    with pytest.raises(RuntimeError, match="HF32 mode is control-flow dependent"):
        _source([flag], body, tmp_path)


@pytest.mark.parametrize("threads", [0, 2049])
def test_native_pto_rejects_invalid_simt_launch(threads, tmp_path):
    thread = I.Var("thread", "int32")
    with pytest.raises((RuntimeError, ValueError), match="positive|2048"):
        _source([], [_block("SIMT_VF", [_thread(thread, threads, [I.Evaluate(I.const(0))])])], tmp_path)


def test_native_pto_pointer_alignment_tracks_nested_offsets(tmp_path):
    gm = _ptr("gm", "float32")
    ub = _ptr("ub", "float32", "shared.dyn")
    shifted = _offset(_offset(ub, 2), 30)
    copy = _call("tl.ascend_copy_ubuf_to_gm", gm, shifted, 0, 1, 32, 0, 0, 0)
    code = _source([gm, ub], [I.Evaluate(copy)], tmp_path)
    assert "pto.mte_ub_gm(" in code
    assert ", 2)" in code and ", 30)" in code


def test_native_pto_dma_retains_declared_arena_view(tmp_path):
    gm = _ptr("gm", "float32")
    arena = _ptr("arena", "uint8", "shared.dyn")
    view = _tensor("view", "float32", "shared.dyn")
    pointer = I.Call("tirx.reinterpret", [_offset(arena, 96)], ty=I.PtrType(I.PrimType("float32"), "shared.dyn"))
    copy = _call("tl.ascend_copy_ubuf_to_gm", gm, _data(view), 0, 1, 32, 0, 0, 0)
    code = _source([gm, arena], [_decl(view, pointer), I.Evaluate(copy)], tmp_path)
    syntax = ast.parse(code)
    assignment = next(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "view"
    )
    assert any(isinstance(node, ast.Name) and node.id == "arena" for node in ast.walk(assignment.value))
    assert any(isinstance(node, ast.Constant) and node.value == 96 for node in ast.walk(assignment.value))
    copy = next(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "mte_ub_gm"
    )
    assert isinstance(copy.args[0], ast.Name) and copy.args[0].id == "view"


def test_native_pto_merging_uses_typed_vector_destination(tmp_path):
    destination = _tensor("result", "float32x64", "local.fragment", (1,))
    mask = _call("tl.simd.pset", 32, "PAT_ALL", dtype="boolx64")
    lhs = _call("tl.simd.vdup", I.const(1, "float32"), mask, "MODE_ZEROING", dtype="float32x64")
    rhs = _call("tl.simd.vdup", I.const(2, "float32"), mask, "MODE_ZEROING", dtype="float32x64")
    merge = _call("tl.simd.vadd", _data(destination), lhs, rhs, mask, "MODE_MERGING")
    code = _source([], [_alloc(destination), I.TensorStore(destination, [0], lhs), I.Evaluate(merge)], tmp_path)
    assert "pto.vsel(" in code and "pto.vadd(" in code
    assert "result_tl_slot_0 = " in code


@pytest.mark.parametrize("offset", [0, 128])
def test_native_pto_packed_l1_copy_retains_arena_base(offset, tmp_path):
    gm = _ptr("gm", "uint8")
    arena = _tensor("l1_arena", "uint8", "shared.l1", (1024,))
    destination = _offset(_data(arena), offset)
    copy = _call("tl.ascend_copy_gm_to_cbuf", destination, gm, 0, 64, 0, 16, 16, 0, 0, 0, 16, "uint16_t")
    code = _source([gm], [_alloc(arena), I.Evaluate(copy)], tmp_path)
    syntax = ast.parse(code)
    call = next(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "mte_gm_l1_frac"
    )
    assert any(isinstance(node, ast.Name) and node.id == "l1_arena" for node in ast.walk(call.args[1]))
    if offset:
        assert any(isinstance(node, ast.Constant) and node.value == offset for node in ast.walk(call.args[1]))


def test_native_pto_packed_l1_copy_checks_byte_alignment(tmp_path):
    gm = _ptr("gm", "uint8")
    arena = _tensor("l1_arena", "uint8", "shared.l1", (1024,))
    copy = _call("tl.ascend_copy_gm_to_cbuf", _offset(_data(arena), 16), gm, 0, 64, 0, 16, 16, 0, 0, 0, 16, "uint16_t")
    with pytest.raises(RuntimeError, match="32-byte aligned"):
        _source([gm], [_alloc(arena), I.Evaluate(copy)], tmp_path)
