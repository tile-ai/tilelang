import ctypes

import pytest
import tvm_ffi

from tvm_tirx import ir, tirx


def packed_forwarder(name, backend, *, copy_slot=False):
    pointer = ir.PtrType(ir.PrimType("void"))
    self_handle = ir.Var("self_handle", pointer)
    arguments = ir.Var("arguments", pointer)
    count = ir.Var("count", "int32")
    result = ir.Var("result", pointer)
    statements = []
    forwarded = arguments
    if copy_slot:
        forwarded = ir.Var("slot", pointer)
        statements.append(ir.Bind(forwarded, ir.Call("tirx.stack_alloca", [ir.StringImm("tvm_ffi_any"), ir.const(1)])))
        for field, dtype in [(13, "int32"), (14, "int32"), (15, pointer)]:
            value = ir.Call("tirx.abi_field_get", [arguments, ir.const(0), ir.const(field)], ty=dtype)
            statements.append(ir.Evaluate(ir.Call("tirx.abi_field_set", [forwarded, ir.const(0), ir.const(field), value])))
        value = ir.Call("tirx.abi_field_get", [forwarded, ir.const(0), ir.const(15)], ty=pointer)
        statements.append(ir.Evaluate(ir.Call("tirx.abi_field_set", [forwarded, ir.const(0), ir.const(15), value])))
    statements.append(
        ir.Return(
            ir.Call(
                "tirx.call_packed",
                [ir.StringImm("tl.runtime.CallGlobalWithResult"), ir.StringImm(name), forwarded, count, result],
                ty="int32",
            )
        )
    )
    function = tirx.Function([self_handle, arguments, count, result], statements, ret_type=ir.PrimType("int32"))
    function = function.with_attr("calling_conv", 1).with_attr("global_symbol", "__tvm_ffi_main").with_attr("tirx.is_entry_func", True)
    return tirx.compile(function, target=backend)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_explicit_result_transfers_object_ownership(backend):
    executable = packed_forwarder("ffi.Array", backend)
    for _ in range(3):
        result = executable(7, "owned result", 2.5)
        assert list(result) == [7, "owned result", 2.5]


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_explicit_result_preserves_error_status(backend):
    name = "tirx.testing.packed_result_failure"

    def fail():
        raise RuntimeError("packed callback failed")

    tvm_ffi.register_global_func(name, fail, override=True)
    try:
        executable = packed_forwarder(name, backend)
        with pytest.raises(RuntimeError, match="packed callback failed"):
            executable()
    finally:
        tvm_ffi.registry.remove_global_func(name)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_union_store_reads_value_before_clearing_same_slot(backend):
    executable = packed_forwarder("ffi.Array", backend, copy_slot=True)
    value = tvm_ffi.get_global_func("ffi.Array")(1, 2, 3)
    result = executable(value)
    assert list(result[0]) == [1, 2, 3]


@pytest.mark.parametrize("registered", [True, False])
def test_raw_c_host_resolves_global_without_module_context(tmp_path, registered):
    from tvm_tirx.codegen import Executable
    from tvm_tirx.target import Target

    name = "tirx.testing.raw_c_host_callback"
    scalar = ir.Var("scalar", "int32")
    result = ir.Call("tirx.call_packed", [ir.StringImm(name), scalar], ty="int32")
    function = tirx.Function(
        [scalar],
        ir.Return(result),
        ret_type=ir.PrimType("int32"),
        attrs=ir.DictAttrs(
            dict={
                "global_symbol": "raw_callback",
                "tirx.is_host_func": True,
                "tirx.is_global_func": True,
                "target": Target("c"),
            }
        ),
    )
    module = tirx.transform.LowerTVMBuiltin()(ir.IRModule({"raw_callback": function}))
    if registered:
        tvm_ffi.register_global_func(name, lambda value: value + 9, override=True)
    try:
        path = tmp_path / "raw_callback.so"
        Executable(tvm_ffi.get_global_func("tl.codegen.build_c")(module, Target("c"))).export_library(path)
        library = ctypes.CDLL(str(path))
        library.raw_callback.argtypes = [ctypes.c_int32]
        library.raw_callback.restype = ctypes.c_int32
        if registered:
            assert library.raw_callback(4) == 13
            assert library.raw_callback(7) == 16
        else:
            from tvm_ffi.libinfo import find_libtvm_ffi

            assert library.raw_callback(4) == -1
            ffi_library = ctypes.CDLL(find_libtvm_ffi())
            ffi_library.TVMFFIErrorMoveFromRaised.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            ffi_library.TVMFFIErrorMoveFromRaised.restype = None
            ffi_library.TVMFFIObjectDecRef.argtypes = [ctypes.c_void_p]
            ffi_library.TVMFFIObjectDecRef.restype = ctypes.c_int
            error = ctypes.c_void_p()
            ffi_library.TVMFFIErrorMoveFromRaised(ctypes.byref(error))
            assert error.value is not None
            assert ffi_library.TVMFFIObjectDecRef(error) == 0
    finally:
        if registered:
            tvm_ffi.registry.remove_global_func(name)
