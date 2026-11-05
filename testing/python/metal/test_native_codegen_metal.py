"""Real native Metal emitter contracts, runnable without an Apple SDK."""

import re

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx.backend.metal import op as M

I = tvm.ir


def _allocation(name, size, dtype, scope):
    tensor = I.Var(name, tvm.tirx.TensorType((size,), dtype, scope=scope, layout="default"))
    return tensor, I.Bind(tensor, I.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _pointer(name, dtype, scope="global"):
    return I.Var(name, I.PtrType(I.PrimType(dtype), scope))


def _thread(body, threads=32):
    var = I.Var("tx", "int32")
    return I.RegionStmt("tirx.launch_thread", [I.StringImm("threadIdx.x"), I.const(threads, "int32")], [var], {}, body)


def _source(params, body, tmp_path, attrs=None):
    func = tvm.tirx.Function(params, body, ret_type=I.PrimType("void"))
    func = func.with_attr("global_symbol", "kernel").with_attr("calling_conv", 2)
    for key, value in (attrs or {}).items():
        func = func.with_attr(key, value)
    source = tvm_ffi.get_global_func("target.build.tilelang_metal_without_compile")(
        tvm.IRModule({"kernel": func}), tvm.target.Target("metal")
    ).inspect_source("metal")
    (tmp_path / "kernel.metal").write_text(source)
    return source


@pytest.mark.parametrize("scope,qualifier", [("shared", "threadgroup"), ("local", "thread")])
def test_native_metal_tensor_allocation_and_access(scope, qualifier, tmp_path):
    buf, allocate = _allocation("values", 32, "float32", scope)
    code = _source([], [_thread([allocate, I.TensorStore(buf, [0], I.const(3, "float32"))])], tmp_path)
    expected_size = 36 if scope == "shared" else 32
    assert f"{qualifier} float values[{expected_size}]" in code
    assert "values[0] = 3.000000e+00f" in code
    assert "max_total_threads_per_threadgroup(32)" in code
    assert "[[thread_position_in_threadgroup]]" in code


def test_native_metal_scalar_parameter_packing_and_no_threads(tmp_path):
    params = [_pointer("data", "float32"), I.Var("count", "int32"), I.Var("wide", "int64"), I.Var("flag", "bool")]
    code = _source(params, [I.Evaluate(params[-1])], tmp_path)
    assert "device float* data [[ buffer(0) ]]" in code
    assert "int count[2]" in code and "long wide;" in code and "int flag[2]" in code
    assert "bool(arg.flag[0])" in code
    assert not re.search(r",\s*\)\s*\{", code)


def test_native_metal_rejects_pointer_after_scalar_parameters(tmp_path):
    params = [I.Var("count", "int32"), _pointer("data", "float32")]
    with pytest.raises(TypeError, match="all pointers before scalar parameters"):
        _source(params, [], tmp_path)


def test_native_metal_pointer_views_keep_element_type_and_address_space(tmp_path):
    source = _pointer("source", "uint8")
    view_type = I.PtrType(I.PrimType("float32"), "global")
    view = I.Var("view", view_type)
    reinterpret = I.Call("tirx.reinterpret", [source], ty=view_type)
    offset = I.Call("tirx.handle_add_byte_offset", [view, I.const(4, "int32")], ty=view_type)
    shifted = I.Var("shifted", view_type)
    code = _source([source], [I.Bind(view, reinterpret), I.Bind(shifted, offset)], tmp_path)
    assert "reinterpret_cast<device float*>(source)" in code
    assert "((device float*)((device char*)view + 4))" in code
    assert "device float* shifted" in code


def test_native_metal_rejects_pointer_address_space_changes(tmp_path):
    source = _pointer("source", "uint8")
    view_type = I.PtrType(I.PrimType("float32"), "shared")
    view = I.Var("view", view_type)
    with pytest.raises(TypeError, match="address space"):
        _source([source], [I.Bind(view, I.Call("tirx.reinterpret", [source], ty=view_type))], tmp_path)


@pytest.mark.parametrize("dtype,spelling", [("float16", "half"), ("float32", "float"), ("bfloat16", "bfloat")])
def test_native_metal_simdgroup_fill_store(dtype, spelling, tmp_path):
    fragment, allocate = _allocation("fragment", 64, dtype, "metal.simdgroup")
    output = _pointer("output", dtype)
    body = [
        allocate,
        I.Evaluate(M.make_filled_simdgroup_matrix(fragment, 0, I.const(1, dtype))),
        I.Evaluate(M.simdgroup_store(fragment, 0, output, 8)),
    ]
    code = _source([output], [_thread(body)], tmp_path)
    assert f"simdgroup_{spelling}8x8 fragment[1]" in code
    assert f"make_filled_simdgroup_matrix<{spelling}, 8, 8>" in code
    assert "simdgroup_store(fragment[0], output, 8, 0, (bool)0)" in code


def _ct_call(name, args):
    call = I.Call("tl.cooperative_tensor_" + name, args, ty_args=[I.PrimType("void")])
    # Native construction infers types; the separate verifier enforces contracts.
    call.validate()
    return call


def test_native_metal_cooperative_tensor_loop_unroll_keeps_allocation_identity(tmp_path):
    fragment, allocate = _allocation("fragment", 1024, "float32", "metal.cooperative_tensor")
    index = I.Var("slot", "int32")
    fill = _ct_call("fill", [fragment, index, I.const(0, "float32"), 16, 32])
    loop = I.For(index, 0, 2, 1, I.ForKind.DEFAULT, [I.Evaluate(fill)])
    code = _source([], [_thread([allocate, loop])], tmp_path)
    assert "MetalPerformancePrimitives" in code
    assert "__pct_c0[__i] = 0.000000e+00f" in code
    assert "__pct_c1[__i] = 0.000000e+00f" in code
    assert "slot" not in code
    assert "thread float fragment[" not in code


def test_native_metal_cooperative_tensor_transfer_and_mma(tmp_path):
    a, aa = _allocation("a", 256, "float16", "metal.cooperative_tensor")
    b, ba = _allocation("b", 512, "float16", "metal.cooperative_tensor")
    c, ca = _allocation("c", 512, "float32", "metal.cooperative_tensor")
    source = _pointer("source", "float16")
    output = _pointer("output", "float32")
    load = _ct_call("load", [a, 0, source, 16, 16, 16, False, 16, 32, 16, 0])
    mma = _ct_call("multiply_accumulate", [c, 0, a, 0, b, 0, c, 0, 16, 32, 16, False, False])
    store = _ct_call("store", [c, 0, output, 32, 16, 32, False, 16, 32, 16, 2])
    code = _source([source, output], [_thread([aa, ba, ca, I.Evaluate(load), I.Evaluate(mma), I.Evaluate(store)])], tmp_path)
    assert "const device half* __src = (const device half*)source" in code
    assert "device float* __dst = (device float*)output" in code
    assert "__pct_op.run(__ct_a, __ct_b, __pct_c0)" in code
    assert "__metal_get_thread_index_in_simdgroup" in code


@pytest.mark.parametrize("invalid", ["wrong_scope", "non_integer_index", "dynamic_dimension"])
def test_cooperative_tensor_rejects_invalid_native_contract(invalid):
    fragment, _ = _allocation("fragment", 512, "float32", "local" if invalid == "wrong_scope" else "metal.cooperative_tensor")
    idx = I.const(0.5, "float32") if invalid == "non_integer_index" else I.const(0, "int32")
    rows = I.Var("rows", "int32") if invalid == "dynamic_dimension" else I.const(16, "int32")
    with pytest.raises((TypeError, ValueError)):
        _ct_call("fill", [fragment, idx, I.const(0, "float32"), rows, 32])


def test_metal_source_only_does_not_load_the_runtime(tmp_path):
    called = []
    name = "ffi.Module.create.metal"
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda *args: called.append(args), override=True)
    try:
        _source([], [], tmp_path)
        assert not called
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, previous, override=True)
