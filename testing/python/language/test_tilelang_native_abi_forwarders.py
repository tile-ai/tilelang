"""Public ABI spellings execute with native typed pointers and owned Any slots."""

import ctypes
import gc
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_tirx import codegen, ir, libinfo, runtime, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.compilation_pipeline import finalize_host_passes

import tilelang
import tilelang.language as T
from tilelang.env import TILELANG_TEMPLATE_PATH


IDS = itertools.count(10000)
VOID = ir.TupleType([])
HANDLE = ir.PtrType("void")
# TileLang owns this ABI envelope. Native C does not consume TL operations,
# and the LLVM route is outside the currently enabled TileLang host pipeline.
BACKENDS = ["tilelang_c"]


def _used_anylist_call(callback, handle, result_type, *arguments, cpacked=False):
    return ir.Call(
        "tirx.call_cpacked" if cpacked else "tirx.call_packed",
        [ir.StringImm(callback), T.anylist_getitem(handle, 0), *arguments],
        ty=result_type,
    )


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("abi_probe")
    source = Path(__file__).resolve().parents[2] / "cpp/native_anylist_probe.cc"
    runtime_lib = libinfo.find_libtvm_runtime()
    module = tvm_ffi.cpp.load(
        "abi106_probe",
        sources=[str(source)],
        extra_cflags=["-std=c++20"],
        extra_include_paths=[libinfo.find_include_path(), TILELANG_TEMPLATE_PATH],
        extra_ldflags=[runtime_lib, "-Wl,-rpath," + str(Path(runtime_lib).parent)],
        build_directory=str(root),
    )
    library = next(root.glob("*.so"))
    # Direct C packed calls resolve the public symbol from this real DSO.
    handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    tvm_ffi.register_global_func("tl.testing.abi106.make", module.abi106_make, override=True)
    tvm_ffi.register_global_func("tl.testing.abi106.echo", module.abi106_echo, override=True)
    tvm_ffi.register_global_func("tl.testing.abi106.mix", module.abi106_mix, override=True)
    yield module, library, handle


def _function(name, params, body):
    return tirx.Function(params, [*body, ir.Return(ir.Tuple([]))], VOID, ir.DictAttrs({"global_symbol": name}))


def _build(function, backend, directory, *, extra_objects=()):
    directory.mkdir(parents=True, exist_ok=True)
    target = Target({"kind": "llvm", "mcpu": "generic"} if backend == "llvm" else "c")
    function = function.with_attr("target", target.with_host(target))
    assert tirx.analysis.check_well_formed(function)
    before = ir.IRModule({function.attrs["global_symbol"]: function})
    lowered = tilelang.transform.MakePackedAPI()(before)
    lowered = tilelang.transform.PrepareAnyList()(lowered)
    lowered = finalize_host_passes()(lowered)
    assert tirx.analysis.check_well_formed(lowered)
    (directory / "input.json").write_text(ir.save_json(before))
    (directory / "lowered.json").write_text(ir.save_json(lowered))
    key = "target.build.llvm" if backend == "llvm" else "target.build.c" if backend == "native_c" else "target.build.tilelang_c_host"
    module = tvm_ffi.get_global_func(key)(lowered, target)
    text = module.inspect_source("ll" if backend == "llvm" else "c")
    (directory / ("generated.ll" if backend == "llvm" else "generated.c")).write_text(text)
    library = directory / "entry.so"
    # Link the selected runtime explicitly, making helper resolution independent
    # of pre-existing global symbol visibility in the test process.
    options = ["-std=c++17", "-I" + libinfo.find_include_path(), "-I" + TILELANG_TEMPLATE_PATH]
    if os.environ.get("TL_TEMPLATE_PATH"):
        options.append("-I" + os.environ["TL_TEMPLATE_PATH"])
    runtime_lib = libinfo.find_libtvm_runtime()
    options += [runtime_lib, "-Wl,-rpath," + str(Path(runtime_lib).parent)]
    if extra_objects:
        # The C packed ABI declaration precedes generated source for external
        # C symbols, exactly as an embedding application's header would.
        header = directory / "external.h"
        header.write_text(
            '#include <tvm/ffi/c_api.h>\nextern "C" {\n'
            "int __tvm_ffi_abi106_make(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);\n"
            "int __tvm_ffi_abi106_echo(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);\n}\n"
            'extern "C" int __tvm_ffi_abi106_mix(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);\n'
        )
        options += ["-include", str(header), *map(str, extra_objects)]
    codegen.Executable(module).export_library(str(library), workspace_dir=str(directory), cc="c++", options=options)
    loaded = runtime.load_module(str(library))
    return loaded[function.attrs["global_symbol"]], library, lowered


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("tvm_stack_alloca", "dtype_str,num"),
        ("tvm_stack_make_shape", "args"),
        ("tvm_stack_make_array", "data,shape,strides,ndim,arr_dtype,elem_offset"),
        ("tvm_throw_last_error", ""),
        ("TVMBackendAllocWorkspace", "device_type,device_id,nbytes,dtype_code_hint,dtype_bits_hint"),
        ("TVMBackendFreeWorkspace", "device_type,device_id,ptr"),
        ("anylist_getitem", "list_handle,index"),
        ("anylist_resetitem", "list_handle,index"),
        ("anylist_setitem_call_packed", "list_handle,index,func_name,args"),
        ("anylist_setitem_call_cpacked", "list_handle,index,func_name,args"),
    ],
)
def test_public_signatures(name, parameters):
    assert ",".join(inspect.signature(getattr(T, name)).parameters) == parameters


def test_native_pointer_and_effect_contracts():
    handle = ir.Var("handle", HANDLE)
    calls = [
        (T.tvm_stack_alloca("shape", 3), "tirx.stack_alloca", ir.PtrType("int64")),
        (T.tvm_stack_alloca("tvm_ffi_any", 2), "tirx.stack_alloca", HANDLE),
        (T.tvm_stack_make_shape(2, 3), "tirx.stack_make_shape", ir.PtrType("int64")),
        (T.tvm_stack_make_array(handle, handle, 0, 1, T.float32(0), 1), "tirx.stack_make_dltensor", HANDLE),
        (T.tvm_throw_last_error(), "tirx.throw_last_error", ir.PrimType("void")),
        (T.TVMBackendAllocWorkspace(1, 0, 128, 2, 32), "tirx.alloc_workspace", HANDLE),
        (T.TVMBackendFreeWorkspace(1, 0, handle), "tirx.free_workspace", ir.PrimType("int32")),
        (T.anylist_getitem(handle, 0), "tl.anylist_getitem", HANDLE),
        (T.anylist_resetitem(handle, 0), "tl.anylist_resetitem", ir.PrimType("int32")),
        (T.anylist_setitem_call_packed(handle, 0, "echo", 3), "tl.anylist_setitem_call_packed", ir.PrimType("int32")),
        (T.anylist_setitem_call_cpacked(handle, 0, "echo", 3), "tl.anylist_setitem_call_cpacked", ir.PrimType("int32")),
    ]
    for call, name, ty in calls:
        assert call.op.name == name and call.ty == ty
        assert len(call.ty_args) == 0
        call.validate()
        assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(call)), call, map_free_vars=True)
        effect = ir.CallEffectKind.ReadState if name == "tl.anylist_getitem" else ir.CallEffectKind.Opaque
        assert int(call.op.get_attr("TCallEffectKind")) == int(effect)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cpacked", [False, True])
def test_mixed_owned_results_and_overwrite(probe, backend, cpacked, tmp_path):
    extension, lib, _ = probe
    handle, kind, ident = ir.Var("list", HANDLE), ir.Var("kind", "int32"), ir.Var("id", "int32")
    op = T.anylist_setitem_call_cpacked if cpacked else T.anylist_setitem_call_packed
    name = "abi106_make" if cpacked else "tl.testing.abi106.make"
    function = _function("set_item", [handle, kind, ident], [ir.Evaluate(op(handle, 1, name, kind, ident))])
    run, _, _ = _build(function, backend, tmp_path / "set", extra_objects=[lib] if cpacked else [])
    arena = extension.new_arena(4)
    pointer = extension.pointer(arena)
    for case in range(7):
        old, new = next(IDS), next(IDS)
        extension.put(arena, 1, extension.abi106_make(6, old))
        run(pointer, case, new)
        assert extension.destroyed(old) == 1
        if case == 6:
            assert extension.tracked_id(arena, 1) == new
            assert extension.destroyed(new) == 0
        else:
            actual = extension.get(arena, 1)
            expected = extension.abi106_make(case, new)
            assert actual == expected and type(actual) is type(expected)
    del arena
    gc.collect()
    assert extension.destroyed(new) == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cpacked", [False, True])
def test_getter_preserves_mixed_types_and_alias_lifetime(probe, backend, cpacked, tmp_path):
    extension, lib, _ = probe
    handle = ir.Var("list", HANDLE)
    op = T.anylist_setitem_call_cpacked if cpacked else T.anylist_setitem_call_packed
    name = "abi106_mix" if cpacked else "tl.testing.abi106.mix"
    body = [ir.Evaluate(op(handle, 1, name, 11, T.float64(2.5), T.bool(True), T.anylist_getitem(handle, 0)))]
    copy, _, _ = _build(_function("copy_item", [handle], body), backend, tmp_path / "copy", extra_objects=[lib] if cpacked else [])
    reset, _, _ = _build(_function("reset_item", [handle], [ir.Evaluate(T.anylist_resetitem(handle, 0))]), backend, tmp_path / "reset")
    arena = extension.new_arena(4)
    pointer = extension.pointer(arena)
    for kind in range(7):
        ident = next(IDS)
        extension.put(arena, 0, extension.abi106_make(kind, ident))
        copy(pointer)
        if kind == 6:
            assert extension.tracked_id(arena, 1) == ident
            reset(pointer)
            assert extension.destroyed(ident) == 0
            extension.reset_raw(pointer, 1)
            assert extension.destroyed(ident) == 1
            extension.reset_raw(pointer, 1)
            assert extension.destroyed(ident) == 1
        else:
            assert extension.get(arena, 1) == extension.get(arena, 0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cpacked", [False, True])
def test_callback_exception_does_not_replace_destination(probe, backend, cpacked, tmp_path):
    extension, lib, _ = probe
    handle = ir.Var("list", HANDLE)
    op = T.anylist_setitem_call_cpacked if cpacked else T.anylist_setitem_call_packed
    name = "abi106_make" if cpacked else "tl.testing.abi106.make"
    body = [ir.Evaluate(op(handle, 0, name, -1, 0))]
    run, _, _ = _build(_function("failing_callback", [handle], body), backend, tmp_path / "fail", extra_objects=[lib] if cpacked else [])
    arena = extension.new_arena(2)
    ident = next(IDS)
    extension.put(arena, 0, extension.abi106_make(6, ident))
    with pytest.raises(ValueError, match="requested callback failure"):
        run(extension.pointer(arena))
    assert extension.tracked_id(arena, 0) == ident
    assert extension.destroyed(ident) == 0
    del arena
    gc.collect()
    assert extension.destroyed(ident) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_packed_calls_reserve_distinct_return_slots(probe, backend, tmp_path):
    extension, _, _ = probe
    handle = ir.Var("list", HANDLE)
    inner = T.anylist_setitem_call_packed(handle, 1, "tl.testing.abi106.make", 1, 4)
    outer = T.anylist_setitem_call_packed(handle, 0, "tl.testing.abi106.echo", inner)
    run, _, lowered = _build(_function("nested", [handle], [ir.Evaluate(outer)]), backend, tmp_path / "nested")
    arena = extension.new_arena(3)
    run(extension.pointer(arena))
    assert extension.get(arena, 0) == 0
    assert extension.get(arena, 1) == 1004
    calls = []
    tvm_ffi.structural_walk(lowered, (ir.Call, calls.append))
    invocations = [x for x in calls if isinstance(x.op, ir.Op) and x.op.name == "tl.anylist_invoke"]
    assert len(invocations) == 2
    generated = (tmp_path / "nested/generated.c").read_text()
    assert "TVMFFIAny anylist_arguments[2]" in generated
    assert "TVMFFIAny anylist_arguments_1[3]" in generated


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("index", [-1, 2147483648, 4294967296])
def test_invalid_dynamic_index_fails_before_touching_storage(probe, backend, index, tmp_path):
    extension, _, _ = probe
    handle, index_var = ir.Var("list", HANDLE), ir.Var("index", "int64")
    body = [ir.Evaluate(T.anylist_setitem_call_packed(handle, index_var, "tl.testing.abi106.make", 1, 2))]
    run, _, _ = _build(_function("invalid_index", [handle, index_var], body), backend, tmp_path / "index")
    arena = extension.new_arena(1)
    with pytest.raises(ValueError, match="index must fit nonnegative int32"):
        run(extension.pointer(arena), index)
    assert extension.get(arena, 0) is None


def test_raw_move_clear_and_borrow_destruction(probe):
    extension, _, _ = probe
    arena = extension.new_arena(2)
    ident = next(IDS)
    extension.move_roundtrip(arena, ident)
    assert extension.get(arena, 0) is None and extension.destroyed(ident) == 1
    with pytest.raises(ValueError, match="must not be null"):
        extension.reset_raw(ctypes.c_void_p(), 0)
    with pytest.raises(ValueError, match="nonnegative"):
        extension.reset_raw(extension.pointer(arena), -1)


def test_getter_marker_cannot_escape_its_call_boundary():
    handle = ir.Var("list", HANDLE)
    function = _function("bad_get", [handle], [ir.Evaluate(T.anylist_getitem(handle, 0))]).with_attr("target", Target("llvm"))
    with pytest.raises(ValueError, match="direct packed-call argument"):
        tilelang.transform.PrepareAnyList()(ir.IRModule({"bad_get": function}))


@pytest.mark.parametrize("op", [T.anylist_getitem, T.anylist_resetitem])
def test_anylist_rejects_nonpointer_and_noninteger_operands(op):
    with pytest.raises(TypeError):
        op(ir.Var("integer", "int64"), 0)
    with pytest.raises(TypeError):
        op(ir.Var("pointer", HANDLE), T.float32(0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_stack_descriptor_offset_strides_and_lifetime(backend, tmp_path):
    handle = ir.Var("data", HANDLE)
    descriptor = T.tvm_stack_make_array(handle, T.tvm_stack_make_shape(2, 3), T.tvm_stack_make_shape(5, 1), 2, T.float32(0), 1)
    captured = []

    def consume(value):
        # Tensor.numpy() intentionally requires compact storage. Gather through
        # the actual descriptor's strides with a separate address oracle.
        assert value.shape == (2, 3) and value.strides == (5, 1)
        assert value.data_ptr() == data.ctypes.data + 4
        memory = (ctypes.c_float * 8).from_address(value.data_ptr())
        captured.append([[memory[i * value.strides[0] + j * value.strides[1]] for j in range(3)] for i in range(2)])

    tvm_ffi.register_global_func("tl.testing.abi106.descriptor", consume, override=True)
    context = ir.RegionStmt(
        "tirx.device_context",
        [ir.const(1), ir.const(0)],
        [],
        None,
        [ir.Evaluate(T.call_packed("tl.testing.abi106.descriptor", descriptor))],
        [],
    )
    function = _function("descriptor", [handle], [context])
    run, _, _ = _build(function, backend, tmp_path / "descriptor")
    data = np.arange(20, dtype="float32")
    run(ctypes.c_void_p(data.ctypes.data))
    np.testing.assert_array_equal(captured, [[[1, 2, 3], [6, 7, 8]]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_workspace_allocate_write_read_free_and_error(backend, tmp_path):
    @T.prim_func
    def workspace():
        ptr = T.TVMBackendAllocWorkspace(1, 0, T.uint64(64), 2, 32)
        if T.isnullptr(ptr):
            T.tvm_throw_last_error()
        view = T.decl_buffer((4,), "float32", data=ptr)
        view[2] = T.float32(17.5)
        T.call_packed("tl.testing.abi106.workspace_value", view[2])
        T.evaluate(T.TVMBackendFreeWorkspace(1, 0, ptr))

    observed = []
    tvm_ffi.register_global_func("tl.testing.abi106.workspace_value", observed.append, override=True)
    run, _, _ = _build(workspace, backend, tmp_path / "workspace")
    run()
    assert observed == [17.5]


def test_public_anylist_source_builds_with_native_ir():
    @T.prim_func
    def program(values: T.handle):
        T.anylist_setitem_call_packed(values, 0, "tl.testing.abi106.echo", T.anylist_getitem(values, 1))
        T.anylist_resetitem(values, 1)

    assert isinstance(program, tirx.Function)
    assert tirx.analysis.check_well_formed(program)
    calls = []
    tvm_ffi.structural_walk(program, (ir.Call, calls.append))
    assert {x.op.name for x in calls if isinstance(x.op, ir.Op)} >= {
        "tl.anylist_setitem_call_packed",
        "tl.anylist_getitem",
        "tl.anylist_resetitem",
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_public_stack_alloca_can_be_written_and_read(backend, tmp_path):
    @T.prim_func
    def stack_values():
        storage = T.tvm_stack_alloca("shape", 2)
        view = T.decl_buffer((2,), "int64", data=storage)
        view[0] = T.int64(29)
        view[1] = T.int64(-17)
        T.call_packed("tl.testing.abi106.stack_values", view[0], view[1])

    observed = []
    tvm_ffi.register_global_func("tl.testing.abi106.stack_values", lambda a, b: observed.append((a, b)), override=True)
    run, _, _ = _build(stack_values, backend, tmp_path / "stack")
    run()
    assert observed == [(29, -17)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_throw_last_error_propagates_workspace_failure(backend, tmp_path):
    @T.prim_func
    def invalid_workspace():
        ptr = T.TVMBackendAllocWorkspace(12, 0, T.uint64(64), 2, 32)
        if T.isnullptr(ptr):
            T.tvm_throw_last_error()
        T.call_packed("tl.testing.abi106.must_not_run")

    called = []
    tvm_ffi.register_global_func("tl.testing.abi106.must_not_run", lambda: called.append(True), override=True)
    run, _, _ = _build(invalid_workspace, backend, tmp_path / "throw")
    with pytest.raises((ValueError, RuntimeError)):
        run()
    assert called == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_null_anylist_storage_propagates_runtime_error(probe, backend, tmp_path):
    handle = ir.Var("list", HANDLE)
    run, _, _ = _build(_function("null_reset", [handle], [ir.Evaluate(T.anylist_resetitem(handle, 0))]), backend, tmp_path / "null")
    with pytest.raises(ValueError, match="must not be null"):
        run(ctypes.c_void_p())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cpacked", [False, True])
def test_anylist_self_alias_retains_single_owner(probe, backend, cpacked, tmp_path):
    extension, library, _ = probe
    handle = ir.Var("list", HANDLE)
    operation = T.anylist_setitem_call_cpacked if cpacked else T.anylist_setitem_call_packed
    callback = "abi106_echo" if cpacked else "tl.testing.abi106.echo"
    body = [ir.Evaluate(operation(handle, 0, callback, T.anylist_getitem(handle, 0)))]
    run, _, _ = _build(
        _function("self_alias", [handle], body), backend, tmp_path / "self_alias", extra_objects=[library] if cpacked else []
    )
    arena = extension.new_arena(1)
    ident = next(IDS)
    extension.put(arena, 0, extension.abi106_make(6, ident))
    for _ in range(3):
        run(extension.pointer(arena))
        assert extension.tracked_id(arena, 0) == ident
        assert extension.destroyed(ident) == 0
    extension.reset_raw(extension.pointer(arena), 0)
    assert extension.destroyed(ident) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_anylist_getter_retains_snapshot_across_later_argument_mutation(probe, backend, tmp_path):
    extension, _, _ = probe
    arena = extension.new_arena(3)
    ident = next(IDS)
    extension.put(arena, 0, extension.abi106_make(6, ident))
    order = []

    def mutate():
        order.append("mutate")
        extension.reset_raw(extension.pointer(arena), 0)
        assert extension.destroyed(ident) == 0
        return 91

    def consume(value, marker):
        order.append("consume")
        assert marker == 91
        extension.put(arena, 2, value)
        assert extension.tracked_id(arena, 2) == ident
        return value

    tvm_ffi.register_global_func("tl.testing.abi106.mutate", mutate, override=True)
    tvm_ffi.register_global_func("tl.testing.abi106.consume_snapshot", consume, override=True)
    handle = ir.Var("list", HANDLE)
    invoke = T.anylist_setitem_call_packed(
        handle, 1, "tl.testing.abi106.consume_snapshot", T.anylist_getitem(handle, 0), T.call_packed("tl.testing.abi106.mutate")
    )
    run, _, _ = _build(_function("snapshot_args", [handle], [ir.Evaluate(invoke)]), backend, tmp_path / "snapshot")
    run(extension.pointer(arena))
    assert order == ["mutate", "consume"]
    assert extension.tracked_id(arena, 1) == ident
    extension.reset_raw(extension.pointer(arena), 1)
    assert extension.destroyed(ident) == 0
    extension.reset_raw(extension.pointer(arena), 2)
    assert extension.destroyed(ident) == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("failure", ["getter", "callback", "lookup"])
def test_anylist_releases_acquired_arguments_on_every_error(probe, backend, failure, tmp_path):
    extension, _, _ = probe
    arena = extension.new_arena(2)
    ident = next(IDS)
    extension.put(arena, 0, extension.abi106_make(6, ident))
    handle = ir.Var("list", HANDLE)
    getter = T.anylist_getitem(handle, 0)
    if failure == "getter":
        later = T.anylist_getitem(handle, -1)
    elif failure == "callback":
        later = T.call_packed("tl.testing.abi106.make", -1, 0)
    else:
        later = T.call_packed("tl.testing.abi106.missing_callback")
    invoke = T.anylist_setitem_call_packed(handle, 1, "tl.testing.abi106.echo", getter, later)
    run, _, _ = _build(_function("error_args", [handle], [ir.Evaluate(invoke)]), backend, tmp_path / "error_args")
    with pytest.raises((ValueError, RuntimeError)):
        run(extension.pointer(arena))
    assert extension.get(arena, 1) is None
    assert extension.destroyed(ident) == 0
    extension.reset_raw(extension.pointer(arena), 0)
    assert extension.destroyed(ident) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_anylist_lazy_branch_does_not_evaluate_unselected_invocation(probe, backend, tmp_path):
    extension, _, _ = probe
    handle = ir.Var("list", HANDLE)
    condition = ir.Var("condition", "bool")
    invoke = T.anylist_setitem_call_packed(handle, 0, "tl.testing.abi106.make", 1, 7)
    value = T.if_then_else(condition, invoke, 0)
    run, _, _ = _build(_function("lazy_args", [handle, condition], [ir.Evaluate(value)]), backend, tmp_path / "lazy")
    run(ctypes.c_void_p(), False)
    arena = extension.new_arena(1)
    run(extension.pointer(arena), True)
    assert extension.get(arena, 0) == 1007


@pytest.mark.parametrize("backend", BACKENDS)
def test_anylist_descriptor_and_string_arguments_use_native_abi(probe, backend, tmp_path):
    extension, _, _ = probe
    input_data = np.arange(8, dtype="float32")
    seen = []

    def observe(tensor, message):
        assert tensor.shape == (2,) and tensor.strides == (2,)
        assert tensor.data_ptr() == input_data.ctypes.data + 4
        values = (ctypes.c_float * 3).from_address(tensor.data_ptr())
        seen.append((values[0], values[2], message))
        return 37

    tvm_ffi.register_global_func("tl.testing.abi106.observe_anylist", observe, override=True)
    handle, data = ir.Var("list", HANDLE), ir.Var("data", HANDLE)
    descriptor = T.tvm_stack_make_array(data, T.tvm_stack_make_shape(2), T.tvm_stack_make_shape(2), 1, T.float32(0), 1)
    invoke = T.anylist_setitem_call_packed(handle, 0, "tl.testing.abi106.observe_anylist", descriptor, "payload")
    context = ir.RegionStmt("tirx.device_context", [ir.const(1), ir.const(0)], [], None, [ir.Evaluate(invoke)], [])
    run, _, _ = _build(_function("descriptor_args", [handle, data], [context]), backend, tmp_path / "descriptor_args")
    arena = extension.new_arena(1)
    run(extension.pointer(arena), ctypes.c_void_p(input_data.ctypes.data))
    assert seen == [(1, 3, "payload")]
    assert extension.get(arena, 0) == 37


@pytest.mark.parametrize("backend", BACKENDS)
def test_anylist_descriptors_keep_distinct_storage_across_nested_native_call(probe, backend, tmp_path):
    extension, _, _ = probe
    input_data = np.arange(20, dtype="float32")
    seen = []

    def gather(tensor):
        memory = (ctypes.c_float * 12).from_address(tensor.data_ptr())
        return [
            memory[sum(i * stride for i, stride in zip(indices, tensor.strides))]
            for indices in itertools.product(*(range(dim) for dim in tensor.shape))
        ]

    def inner(tensor):
        assert tensor.shape == (4,) and tensor.strides == (1,)
        assert tensor.data_ptr() == input_data.ctypes.data + 12
        seen.append(("inner", gather(tensor)))
        return 17

    def outer(first, second, inner_result):
        assert first.shape == (2, 3) and first.strides == (5, 1)
        assert second.shape == (3, 2) and second.strides == (1, 4)
        assert first.data_ptr() == input_data.ctypes.data + 4
        assert second.data_ptr() == input_data.ctypes.data + 8
        assert inner_result == 17
        seen.append(("outer", gather(first), gather(second)))
        return 29

    tvm_ffi.register_global_func("tl.testing.abi106.descriptor_inner", inner, override=True)
    tvm_ffi.register_global_func("tl.testing.abi106.descriptor_outer", outer, override=True)
    handle, data = ir.Var("list", HANDLE), ir.Var("data", HANDLE)
    first = T.tvm_stack_make_array(data, T.tvm_stack_make_shape(2, 3), T.tvm_stack_make_shape(5, 1), 2, T.float32(0), 1)
    second = T.tvm_stack_make_array(data, T.tvm_stack_make_shape(3, 2), T.tvm_stack_make_shape(1, 4), 2, T.float32(0), 2)
    nested = T.tvm_stack_make_array(data, T.tvm_stack_make_shape(4), T.tvm_stack_make_shape(1), 1, T.float32(0), 3)
    invoke = T.anylist_setitem_call_packed(
        handle,
        0,
        "tl.testing.abi106.descriptor_outer",
        first,
        second,
        T.call_packed("tl.testing.abi106.descriptor_inner", nested),
    )
    context = ir.RegionStmt("tirx.device_context", [ir.const(1), ir.const(0)], [], None, [ir.Evaluate(invoke)], [])
    run, _, _ = _build(_function("distinct_descriptors", [handle, data], [context]), backend, tmp_path / "distinct_descriptors")
    arena = extension.new_arena(1)
    for increment in (0, 100):
        input_data[:] = np.arange(20, dtype="float32") + increment
        run(extension.pointer(arena), ctypes.c_void_p(input_data.ctypes.data))
        assert seen[-2:] == [
            ("inner", [increment + i for i in (3, 4, 5, 6)]),
            ("outer", [increment + i for i in (1, 2, 3, 6, 7, 8)], [increment + i for i in (2, 6, 3, 7, 4, 8)]),
        ]
        assert extension.get(arena, 0) == 29


@pytest.mark.parametrize("backend", BACKENDS)
def test_anylist_export_reload_in_fresh_process(probe, backend, tmp_path):
    extension, probe_lib, _ = probe
    handle = ir.Var("list", HANDLE)
    body = [ir.Evaluate(T.anylist_setitem_call_packed(handle, 0, "tl.testing.abi106.echo", T.anylist_getitem(handle, 1)))]
    _, library, _ = _build(_function("reload_copy", [handle], body), backend, tmp_path / "reload")
    child = tmp_path / "reload.py"
    child.write_text("""import gc, hashlib, json, sys
from pathlib import Path
import tvm_ffi, tvm_tirx
from tvm_tirx import runtime
probe = tvm_ffi.load_module(sys.argv[2])
tvm_ffi.register_global_func("tl.testing.abi106.echo", probe.abi106_echo, override=True)
arena = probe.new_arena(3)
probe.put(arena, 1, probe.abi106_make(6, 777001))
module = runtime.load_module(sys.argv[1])
module["reload_copy"](probe.pointer(arena))
assert probe.tracked_id(arena, 0) == 777001
probe.reset_raw(probe.pointer(arena), 1)
assert probe.destroyed(777001) == 0
probe.reset_raw(probe.pointer(arena), 0)
assert probe.destroyed(777001) == 1
assert "tvm" not in sys.modules
paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/" in line}
selected = [p for p in paths if Path(p).name in ("libtvm_ffi.so", "libtvm_runtime.so", "libtvm_tirx_compiler.so")]
assert len([p for p in selected if Path(p).name == "libtvm_ffi.so"]) == 1
assert len([p for p in selected if Path(p).name == "libtvm_runtime.so"]) == 1
assert all("/3rdparty/tvm/" not in p for p in paths)
print(json.dumps({"status":"passed", "loaded":[{"path":p,"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in selected]},sort_keys=True))
""")
    result = subprocess.run(
        [sys.executable, "-B", str(child), str(library), str(probe_lib)], env=dict(os.environ), text=True, capture_output=True
    )
    (tmp_path / "reload.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "passed"' in result.stdout


ANYLIST_OPS = [
    "anylist_getitem",
    "anylist_resetitem",
    "anylist_setitem_call_packed",
    "anylist_setitem_call_cpacked",
]


def _raw_anylist_inputs(name):
    handle = ir.Var("storage", HANDLE)
    arguments = [handle, ir.const(0)]
    if "call" in name:
        arguments += [ir.StringImm("target")]
    result_type = HANDLE if name == "anylist_getitem" else ir.PrimType("int32")
    return arguments, result_type


@pytest.mark.parametrize("name", ANYLIST_OPS)
def test_explicitly_typed_anylist_calls_validate(name):
    arguments, result_type = _raw_anylist_inputs(name)
    ir.Call("tl." + name, arguments, ty=result_type).validate()


@pytest.mark.parametrize("name", ANYLIST_OPS)
@pytest.mark.parametrize("case", ["wrong_result", "nonpointer", "vector_index", "attrs", "type_args", "missing_args"])
def test_raw_anylist_validator_rejects_malformed_calls(name, case):
    arguments, result_type = _raw_anylist_inputs(name)
    options = {"ty": result_type}
    if case == "wrong_result":
        options["ty"] = ir.PrimType("float32")
    elif case == "nonpointer":
        arguments[0] = ir.const(1)
    elif case == "vector_index":
        arguments[1] = ir.Var("lanes", "int32x4")
    elif case == "attrs":
        options["attrs"] = ir.DictAttrs({"ignored": 1})
    elif case == "type_args":
        options["ty_args"] = [ir.PrimType("int32")]
    elif case == "missing_args":
        arguments = []
    with pytest.raises((ValueError, TypeError)):
        ir.Call("tl." + name, arguments, **options).validate()


DESCRIPTOR_CASES = [
    pytest.param((), None, 3, id="rank0"),
    pytest.param((2, 3), (3, 1), 0, id="explicit-compact"),
    pytest.param((2, 3), (5, 1), 1, id="explicit-noncompact"),
    pytest.param((2, 3), None, 2, id="implicit-compact-offset"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape,strides,offset", DESCRIPTOR_CASES)
def test_stack_descriptor_raw_fields(probe, backend, shape, strides, offset, tmp_path):
    # Direct packed-call operands are consumed while their descriptor is live.
    # Nested packed-call return expressions have separate legacy limitations.
    extension, _, _ = probe
    tvm_ffi.register_global_func("tl.testing.abi154.observe", extension.abi154_observe, override=True)
    handle = ir.Var("data", HANDLE)
    descriptor = T.tvm_stack_make_array(
        handle,
        T.tvm_stack_make_shape(*shape),
        0 if strides is None else T.tvm_stack_make_shape(*strides),
        len(shape),
        T.float32(0),
        offset,
    )
    call = T.call_packed("tl.testing.abi154.observe", descriptor, 0)
    context = ir.RegionStmt("tirx.device_context", [ir.const(1), ir.const(0)], [], None, [ir.Evaluate(call)], [])
    run, _, _ = _build(_function("descriptor_direct", [handle], [context]), backend, tmp_path / "descriptor_direct")
    input_data = np.arange(64, dtype="float32")
    observations = []
    for increment in [0, 100]:
        input_data[:] = np.arange(64, dtype="float32") + increment
        extension.abi154_clear()
        run(ctypes.c_void_p(input_data.ctypes.data))
        records = list(extension.abi154_records())
        raw_records = [
            [int(value) for value in record[:12]] + [list(record[12]), None if record[13] is None else list(record[13]), list(record[14])]
            for record in records
        ]
        (tmp_path / f"raw-descriptor-observations-{increment}.json").write_text(json.dumps(raw_records, indent=2))
        assert len(records) == 1
        record = records[0]
        header = [int(value) for value in record[:12]]
        actual_shape = tuple(record[12])
        actual_steps = None if record[13] is None else tuple(record[13])
        actual_values = list(record[14])
        assert header[0] == 0 and header[1] != 0 and header[2] == input_data.ctypes.data
        assert header[5:12] == [len(shape), 1, 0, 2, 32, 1, offset * 4]
        assert actual_shape == shape
        assert bool(header[4]) == (strides is not None)
        assert actual_steps == strides
        if shape:
            assert header[3] != 0
        effective_steps = strides
        if effective_steps is None:
            effective_steps = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
        expected = [
            float(input_data[offset + sum(index * stride for index, stride in zip(indices, effective_steps, strict=True))])
            for indices in itertools.product(*(range(dim) for dim in shape))
        ]
        assert actual_values == expected
        observations.append({"input_increment": increment, "header": header, "shape": shape, "strides": strides, "values": actual_values})
    (tmp_path / "descriptor-observations.json").write_text(json.dumps(observations, indent=2))


@pytest.mark.parametrize("cpacked", [False, True])
@pytest.mark.parametrize("kind", [4, 5, 6])
def test_used_pointer_result_rejects_owned_values_without_losing_the_owner(probe, cpacked, kind, tmp_path):
    extension, library, _ = probe
    ident = next(IDS)
    handle, result = ir.Var("list", HANDLE), ir.Var("result", HANDLE)
    callback = "abi106_make" if cpacked else "tl.testing.abi106.make"
    value = _used_anylist_call(callback, handle, HANDLE, ir.const(ident, "int64"), cpacked=cpacked)
    observed = []

    def observe(pointer):
        # Never dereference a potentially freed pointer. The original unsafe
        # candidate exposes the pointer after the unique owner was destroyed.
        observed.append((pointer.value if pointer is not None else None, extension.destroyed(ident)))

    tvm_ffi.register_global_func("tl.testing.abi106.pointer_after_call", observe, override=True)
    body = [ir.Bind(result, value), ir.Evaluate(T.call_packed("tl.testing.abi106.pointer_after_call", result))]
    run, _, _ = _build(
        _function("used_pointer", [handle], body), "tilelang_c", tmp_path / "used_pointer", extra_objects=[library] if cpacked else []
    )
    arena = extension.new_arena(1)
    extension.put(arena, 0, kind)
    error = None
    try:
        run(extension.pointer(arena))
    except TypeError as caught:
        error = str(caught)
    gc.collect()
    facts = {"observed_after_call": observed, "destructions": extension.destroyed(ident), "type_error": error}
    (tmp_path / "owned-result-observation.json").write_text(json.dumps(facts, indent=2))
    assert error is not None and "Any-list packed result" in error
    assert observed == []
    if kind == 6:
        assert extension.destroyed(ident) == 1


@pytest.mark.parametrize("result_type", [HANDLE, "int64", "float64", "bool"])
def test_used_result_type_error_releases_new_object_and_acquired_argument(probe, result_type, tmp_path):
    extension, _, _ = probe
    input_id, output_id = next(IDS), next(IDS)
    arena = extension.new_arena(1)
    extension.put(arena, 0, extension.abi106_make(6, input_id))
    result_type = ir.PrimType(result_type) if isinstance(result_type, str) else result_type
    handle, result = ir.Var("list", HANDLE), ir.Var("result", result_type)
    tvm_ffi.register_global_func("tl.testing.abi106.fresh_result", lambda _input: extension.abi106_make(6, output_id), override=True)
    value = _used_anylist_call("tl.testing.abi106.fresh_result", handle, result_type)
    observed = []
    tvm_ffi.register_global_func("tl.testing.abi106.must_not_observe_result", observed.append, override=True)
    body = [ir.Bind(result, value), ir.Evaluate(T.call_packed("tl.testing.abi106.must_not_observe_result", result))]
    run, _, _ = _build(_function("bad_result_type", [handle], body), "tilelang_c", tmp_path / "bad_result_type")
    with pytest.raises(TypeError, match="Any-list packed result"):
        run(extension.pointer(arena))
    assert observed == [] and extension.destroyed(output_id) == 1
    assert extension.destroyed(input_id) == 0
    extension.reset_raw(extension.pointer(arena), 0)
    assert extension.destroyed(input_id) == 1


def test_used_pointer_result_rejects_tensor_and_releases_dlpack_owner(probe, tmp_path):
    extension, _, _ = probe
    released = []

    def make_tensor(_input):
        data = np.arange(4, dtype="float32")
        weakref.finalize(data, released.append, "tensor_owner")
        return tvm_ffi.from_dlpack(data)

    tvm_ffi.register_global_func("tl.testing.abi106.fresh_tensor_result", make_tensor, override=True)
    handle, result = ir.Var("list", HANDLE), ir.Var("result", HANDLE)
    observed = []
    tvm_ffi.register_global_func("tl.testing.abi106.must_not_observe_tensor", observed.append, override=True)
    value = _used_anylist_call("tl.testing.abi106.fresh_tensor_result", handle, HANDLE)
    body = [ir.Bind(result, value), ir.Evaluate(T.call_packed("tl.testing.abi106.must_not_observe_tensor", result))]
    run, _, _ = _build(_function("bad_tensor_result", [handle], body), "tilelang_c", tmp_path / "bad_tensor_result")
    arena = extension.new_arena(1)
    extension.put(arena, 0, 0)
    with pytest.raises(TypeError, match="Any-list packed result"):
        run(extension.pointer(arena))
    gc.collect()
    assert released == ["tensor_owner"] and observed == []


@pytest.mark.parametrize(
    "result_type,returned,expected",
    [
        (HANDLE, None, None),
        (HANDLE, ctypes.c_void_p(0x1234), 0x1234),
        ("int64", -7, -7),
        ("int64", True, 1),
        ("bool", 2, True),
        ("float64", 3.25, 3.25),
        ("float64", -7, -7.0),
        ("float64", True, 1.0),
    ],
)
def test_used_scalar_and_opaque_results_follow_ffi_cast_contract(probe, result_type, returned, expected, tmp_path):
    extension, _, _ = probe
    result_type = ir.PrimType(result_type) if isinstance(result_type, str) else result_type
    handle, result = ir.Var("list", HANDLE), ir.Var("result", result_type)
    tvm_ffi.register_global_func("tl.testing.abi106.typed_result", lambda _input: returned, override=True)
    observed = []

    def observe(value):
        observed.append(value.value if isinstance(value, ctypes.c_void_p) else value)

    tvm_ffi.register_global_func("tl.testing.abi106.observe_typed_result", observe, override=True)
    value = _used_anylist_call("tl.testing.abi106.typed_result", handle, result_type)
    body = [ir.Bind(result, value), ir.Evaluate(T.call_packed("tl.testing.abi106.observe_typed_result", result))]
    run, _, _ = _build(_function("typed_result", [handle], body), "tilelang_c", tmp_path / "typed_result")
    arena = extension.new_arena(1)
    extension.put(arena, 0, 0)
    run(extension.pointer(arena))
    assert observed == [expected]


@pytest.mark.parametrize("payload", [-1, -(1 << 63), (1 << 63) - 1])
def test_used_uint64_result_preserves_ffi_int_payload_bits(probe, payload, tmp_path):
    extension, _, _ = probe
    handle = ir.Var("list", HANDLE)
    # NumPy's borrowed storage does not promise native TensorType's default
    # 64-byte alignment. This scalar store only requires the actual 1-byte ABI.
    output = ir.Var("output", tirx.TensorType([1], "uint64", align=1))
    tvm_ffi.register_global_func("tl.testing.abi106.uint64_result", lambda _input: payload, override=True)
    value = _used_anylist_call("tl.testing.abi106.uint64_result", handle, ir.PrimType("uint64"))
    run, _, _ = _build(
        _function("uint64_result", [handle, output], [ir.TensorStore(output, [0], value)]), "tilelang_c", tmp_path / "uint64_result"
    )
    arena = extension.new_arena(1)
    extension.put(arena, 0, 0)
    actual = np.zeros(1, dtype="uint64")
    run(extension.pointer(arena), tvm_ffi.from_dlpack(actual))
    assert int(actual[0]) == payload % (1 << 64)


@pytest.mark.parametrize("return_object", [False, True])
def test_discarded_public_call_result_keeps_none_and_releases_owned_value(probe, return_object, tmp_path):
    extension, _, _ = probe
    ident = next(IDS)
    handle = ir.Var("list", HANDLE)
    observed = []

    def callback(value):
        observed.append(value)
        return extension.abi106_make(6, ident) if return_object else None

    tvm_ffi.register_global_func("tl.testing.abi106.discard_result", callback, override=True)
    call = T.call_packed("tl.testing.abi106.discard_result", T.anylist_getitem(handle, 0))
    run, _, lowered = _build(_function("discard_result", [handle], [ir.Evaluate(call)]), "tilelang_c", tmp_path / "discard_result")
    arena = extension.new_arena(1)
    extension.put(arena, 0, 37)
    run(extension.pointer(arena))
    assert observed == [37]
    assert extension.destroyed(ident) == int(return_object)
    invocations = []
    tvm_ffi.structural_walk(lowered, (ir.Call, invocations.append))
    assert any(call.op.name == "tl.anylist_invoke" and call.ty == ir.PrimType("void") for call in invocations if isinstance(call.op, ir.Op))


@pytest.mark.parametrize("invalid_return", [False, True])
def test_nested_used_result_in_setter_keeps_type_and_cleans_error(probe, invalid_return, tmp_path):
    extension, _, _ = probe
    input_id, output_id = next(IDS), next(IDS)
    arena = extension.new_arena(2)
    extension.put(arena, 0, extension.abi106_make(6, input_id))
    extension.put(arena, 1, 11)

    def inner(_input):
        return extension.abi106_make(6, output_id) if invalid_return else 37

    tvm_ffi.register_global_func("tl.testing.abi106.nested_used_result", inner, override=True)
    handle = ir.Var("list", HANDLE)
    value = _used_anylist_call("tl.testing.abi106.nested_used_result", handle, ir.PrimType("int64"))
    setter = T.anylist_setitem_call_packed(handle, 1, "tl.testing.abi106.echo", value)
    run, _, lowered = _build(
        _function("nested_used_result", [handle], [ir.Evaluate(setter)]), "tilelang_c", tmp_path / "nested_used_result"
    )
    if invalid_return:
        with pytest.raises(TypeError, match="Any-list packed result"):
            run(extension.pointer(arena))
        assert extension.get(arena, 1) == 11
        assert extension.destroyed(output_id) == 1
    else:
        run(extension.pointer(arena))
        assert extension.get(arena, 1) == 37
    assert extension.destroyed(input_id) == 0
    extension.reset_raw(extension.pointer(arena), 0)
    assert extension.destroyed(input_id) == 1
    calls = []
    tvm_ffi.structural_walk(lowered, (ir.Call, calls.append))
    nested = [
        call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tl.anylist_invoke" and call.attrs["mode"] == "value"
    ]
    assert len(nested) == 1 and nested[0].ty == ir.PrimType("int64")
