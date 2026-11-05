"""Packed callbacks retain helper returns and the owning function's stack."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT, SourceHostProgram
from test_native_source_host_packed import build_source, packed


def _compile_helper(helper, *, roundtrip):
    n = ir.Var("n", "int64")
    output = ir.Var("output", tirx.TensorType([1], "int64"))
    helper_global = ir.GlobalVar("helper")
    entry_global = ir.GlobalVar("main")
    entry = tirx.Function(
        [n, output],
        [ir.TensorStore(output, [ir.const(0)], ir.Call(helper_global, [n], ty="int64"))],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "main", "target": Target("c")}),
    )
    graph = ir.IRModule({helper_global: helper, entry_global: entry})
    host = graph.with_attr("tl.source_host_schema", 2).with_attr("tl.source_host_module", graph)
    if roundtrip:
        entry, host = (ir.load_json(ir.save_json(value)) for value in (entry, host))
    assert tirx.analysis.check_well_formed(host)
    source = NATIVE_SOURCE_INIT + SourceHostProgram(entry, host).emit(include_preamble=True)
    generator = LibraryGenerator(Target("c"))
    generator.assign_pass_configs({})
    generator.assign_compile_flags(["-O0", "-fno-inline"])
    generator.update_lib_code(source)
    generator.compile_lib()
    library = generator.load_lib()
    library.call.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    library.call.restype = ctypes.c_int32
    library.get_last_error.restype = ctypes.c_char_p
    return library


@pytest.mark.parametrize("roundtrip", [False, True])
def test_packed_helper_return_keeps_argument_order_and_exception(roundtrip):
    name = "tl.testing.source_packed_helper_return_" + str(roundtrip)
    seen = []

    def left(value):
        seen.append("left")
        if value == -7:
            raise ValueError("helper callback failed before destination store")
        return value + 1

    def right(value):
        seen.append("right")
        return value + 2

    def add(a, b):
        seen.append("sum")
        return a + b

    for suffix, callback in [("_left", left), ("_right", right), ("_sum", add)]:
        tvm_ffi.register_global_func(name + suffix, callback, override=True)
    n = ir.Var("n", "int64")
    conditional = ir.Call("prim.if_then_else", [n > 0, packed(name + "_right", n), ir.const(0, "int64")], ty="int64")
    result = packed(name + "_sum", packed(name + "_left", n), conditional)
    helper = tirx.Function([n], [ir.Return(result)], ir.PrimType("int64"), ir.DictAttrs({"target": Target("c")}))
    library = _compile_helper(helper, roundtrip=roundtrip)
    output = ctypes.c_int64(-123)
    assert library.call(3, ctypes.byref(output), None) == 0
    assert output.value == 9 and seen == ["left", "right", "sum"]
    seen.clear()
    output.value = -123
    assert library.call(-7, ctypes.byref(output), None) == -1
    assert output.value == -123 and seen == ["left"]
    assert b"helper callback failed before destination store" in library.get_last_error()


@pytest.mark.parametrize("roundtrip", [False, True])
def test_packed_helper_stack_alias_keeps_owning_function_lifetime(roundtrip):
    _check_stack_alias(in_helper=True, roundtrip=roundtrip)


def test_packed_entry_stack_alias_outlives_callback_lambda():
    _check_stack_alias(in_helper=False, roundtrip=False)


def _check_stack_alias(*, in_helper, roundtrip):
    name = f"tl.testing.source_packed_stack_scope_{in_helper}_{roundtrip}"
    addresses = []

    def initialize(pointer, value):
        addresses.append(pointer.value)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64))[0] = value
        return pointer

    def overwrite_other(pointer, value):
        addresses.append(pointer.value)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64))[0] = value + 1000
        return 0

    tvm_ffi.register_global_func(name + "_init", initialize, override=True)
    tvm_ffi.register_global_func(name + "_other", overwrite_other, override=True)
    n = ir.Var("n", "int64")
    storage = ir.Var("storage", ir.PtrType("void"))
    other = ir.Var("other_result", "int64")
    view = ir.Var("view", tirx.TensorType([1], "int64"))

    def allocate():
        return ir.Call("tirx.stack_alloca", [ir.StringImm("shape"), ir.const(2, "int32")], ty=ir.PtrType("void"))

    body = [
        ir.Bind(storage, packed(name + "_init", allocate(), n, dtype=storage.ty)),
        ir.Bind(other, packed(name + "_other", allocate(), n)),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [storage], ty_args=[view.ty])),
    ]
    if in_helper:
        helper = tirx.Function([n], [*body, ir.Return(view[0])], ir.PrimType("int64"), ir.DictAttrs({"target": Target("c")}))
        library = _compile_helper(helper, roundtrip=roundtrip)
    else:
        library, _ = build_source(view[0], n, prefix=body)
    output = ctypes.c_int64()
    for value in [31, 92]:
        addresses.clear()
        assert library.call(value, ctypes.byref(output), None) == 0
        assert len(addresses) == 2 and addresses[0] != addresses[1]
        assert output.value == value
