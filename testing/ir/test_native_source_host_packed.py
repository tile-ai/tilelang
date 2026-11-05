"""Source libraries invoke real FFI callbacks without a Module import context."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT, SourceHostProgram


def packed(name, *args, dtype="int64"):
    return ir.Call("tirx.call_packed", [ir.StringImm(name), *args], ty=dtype)


def build_source(expression, n, dtype="int64", prefix=()):
    pointer = ir.Var("output", ir.PtrType(dtype))
    tensor = ir.Var("output_view", tirx.TensorType([1], dtype))
    function = tirx.Function(
        [n, pointer],
        [
            ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty])),
            *prefix,
            ir.TensorStore(tensor, [ir.const(0, "int32")], expression),
        ],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "main", "target": Target("c")}),
    )
    graph = ir.IRModule({"main": function})
    module = graph.with_attr("tl.source_host_schema", 2).with_attr("tl.source_host_module", graph)
    assert tirx.analysis.check_well_formed(module)
    source = NATIVE_SOURCE_INIT + SourceHostProgram(function, module).emit(include_preamble=True)
    generator = LibraryGenerator(Target("c"))
    generator.assign_pass_configs({})
    generator.assign_compile_flags([])
    generator.update_lib_code(source)
    generator.compile_lib()
    library = generator.load_lib()
    library.call.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    library.call.restype = ctypes.c_int32
    library.get_last_error.restype = ctypes.c_char_p
    return library, source


@pytest.mark.parametrize(
    "dtype,ctype,value",
    [("int64", ctypes.c_int64, 23), ("float32", ctypes.c_float, 1.25), ("bool", ctypes.c_bool, True)],
)
def test_source_packed_scalar_result_is_consumed(dtype, ctype, value):
    name = f"tl.testing.source_packed_scalar_{dtype}"
    seen = []

    def callback(n, text):
        seen.append((n, text))
        return value

    tvm_ffi.register_global_func(name, callback, override=True)
    n = ir.Var("n", "int64")
    library, source = build_source(packed(name, n, ir.StringImm("payload"), dtype=dtype), n, dtype)
    output = ctype()
    assert library.call(17, ctypes.byref(output), None) == 0
    assert output.value == value
    assert seen == [(17, "payload")]
    assert "TVMFFIEnvModLookupFromImports(" not in source


def test_source_packed_arguments_evaluate_once_in_order():
    seen = []
    prefix = "tl.testing.source_packed_order_"

    def record(name, value):
        seen.append(name)
        return value

    tvm_ffi.register_global_func(prefix + "left", lambda n: record("left", n + 1), override=True)
    tvm_ffi.register_global_func(prefix + "right", lambda n: record("right", n + 2), override=True)
    tvm_ffi.register_global_func(prefix + "sum", lambda x, y: record("sum", x + y), override=True)
    n = ir.Var("n", "int64")
    expression = packed(prefix + "sum", packed(prefix + "left", n), packed(prefix + "right", n))
    library, _ = build_source(expression, n)
    output = ctypes.c_int64()
    for value in [3, 10]:
        seen.clear()
        assert library.call(value, ctypes.byref(output), None) == 0
        assert output.value == value * 2 + 3
        assert seen == ["left", "right", "sum"]


@pytest.mark.parametrize("discard_result", [False, True])
def test_source_packed_conditional_argument_preserves_order(discard_result):
    seen = []
    prefix = "tl.testing.source_packed_conditional_"

    def record(name, value):
        seen.append(name)
        return value

    tvm_ffi.register_global_func(prefix + "left", lambda n: record("left", n + 1), override=True)
    tvm_ffi.register_global_func(prefix + "right", lambda n: record("right", n + 2), override=True)
    tvm_ffi.register_global_func(prefix + "sum", lambda x, y: record("sum", x + y), override=True)
    n = ir.Var("n", "int64")
    conditional = ir.Call("prim.if_then_else", [n > 0, packed(prefix + "right", n), ir.const(0, "int64")])
    expression = packed(prefix + "sum", packed(prefix + "left", n), conditional)
    library, _ = build_source(
        n if discard_result else expression,
        n,
        prefix=[ir.Evaluate(expression)] if discard_result else [],
    )
    for value in [3, -2]:
        seen.clear()
        output = ctypes.c_int64()
        assert library.call(value, ctypes.byref(output), None) == 0
        expected = value + 1 + (value + 2 if value > 0 else 0)
        assert output.value == (value if discard_result else expected)
        assert seen == (["left", "right", "sum"] if value > 0 else ["left", "sum"])


def test_source_packed_discarded_none_result_keeps_effect():
    name = "tl.testing.source_packed_discarded_none"
    seen = []
    tvm_ffi.register_global_func(name, lambda n: seen.append(n), override=True)
    n = ir.Var("n", "int64")
    library, _ = build_source(n, n, prefix=[ir.Evaluate(packed(name, n))])
    output = ctypes.c_int64()
    assert library.call(9, ctypes.byref(output), None) == 0
    assert output.value == 9 and seen == [9]


@pytest.mark.parametrize("failure", ["missing", "exception", "return_type"])
def test_source_packed_errors_stop_before_output_store(failure):
    name = f"tl.testing.source_packed_failure_{failure}"
    if failure == "exception":

        def fail(n):
            raise ValueError("source callback negative oracle")

        tvm_ffi.register_global_func(name, fail, override=True)
    elif failure == "return_type":
        tvm_ffi.register_global_func(name, lambda n: None, override=True)
    else:
        assert tvm_ffi.get_global_func(name, allow_missing=True) is None
    n = ir.Var("n", "int64")
    library, _ = build_source(packed(name, n), n)
    output = ctypes.c_int64(-123)
    assert library.call(7, ctypes.byref(output), None) == -1
    assert output.value == -123
    error = library.get_last_error().decode()
    if failure == "exception":
        assert "source callback negative oracle" in error
    elif failure == "missing":
        assert name in error
    else:
        assert "int" in error and "None" in error
    # The source adapter keeps its error text until init or the next call.
    assert library.get_last_error().decode() == error
    assert library.init() == 0
    assert library.get_last_error() == b""


def test_source_packed_success_clears_unread_error():
    name = "tl.testing.source_packed_unread_error"

    def callback(n):
        if n < 0:
            raise ValueError("unread source callback error")
        return n + 4

    tvm_ffi.register_global_func(name, callback, override=True)
    n = ir.Var("n", "int64")
    library, _ = build_source(packed(name, n), n)
    output = ctypes.c_int64(-123)
    assert library.call(-1, ctypes.byref(output), None) == -1
    assert output.value == -123
    assert library.call(7, ctypes.byref(output), None) == 0
    assert output.value == 11
    assert library.get_last_error() == b""
