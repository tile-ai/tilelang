"""Native Python host statements use real FFI callbacks and typed arithmetic."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

from tilelang.errors import InternalError
from tilelang.jit.adapter.device_py_host import NativePythonHost, HostResources


def retained_host(function, helpers=None):
    global_var = ir.GlobalVar("main")
    functions = {global_var: function, **(helpers or {})}
    return ir.IRModule(functions).with_attr("tl.source_host_schema", 2).with_attr("tl.source_host_module", ir.IRModule(functions))


def host_function(parameters, statements):
    return tirx.Function(parameters, statements, ir.PrimType("void"), ir.DictAttrs({"global_symbol": "main"}))


def compile_host(function, module=None):
    emitter = NativePythonHost(function, module or retained_host(function), ir.IRModule({}))
    source = emitter.emit()
    namespace = {}
    exec(compile(source, "native-host.py", "exec"), namespace)
    return namespace["call"], source


@pytest.mark.parametrize("roundtrip", [False, True])
def test_nested_control_flow_and_same_name_variables(roundtrip):
    seen = []
    tvm_ffi.register_global_func("tilelang.testing.native_host_record", lambda value: seen.append(value), override=True)
    first = ir.Var("same_name", "int32")
    second = ir.Var("same_name", "int32")
    index = ir.Var("same_name", "int32")
    record = tirx.call_packed("tilelang.testing.native_host_record", first + index * second, ty="void")
    function = host_function(
        [first, second],
        [
            ir.For(
                index,
                ir.const(0),
                ir.const(5),
                ir.const(1),
                ir.ForKind.DEFAULT,
                [
                    ir.If(index == 1, [ir.Continue()], []),
                    ir.If(index == 4, [ir.Break()], []),
                    ir.Evaluate(record),
                ],
            ),
            ir.Return(ir.Tuple([])),
        ],
    )
    module = retained_host(function)
    if roundtrip:
        function, module = ir.load_json(ir.save_json(function)), ir.load_json(ir.save_json(module))
    call, source = compile_host(function, module)
    assert call({}, 7, 11) == 0
    assert seen == [7, 29, 40]
    assert "for " in source and "continue" in source and "break" in source


@pytest.mark.parametrize("left,right", [(-7, 3), (7, -3), (-7, -3), (2**54 + 3, 7)])
def test_floor_and_truncating_division_have_distinct_exact_oracles(left, right):
    seen = []
    tvm_ffi.register_global_func("tilelang.testing.native_host_arith", lambda *args: seen.append(args), override=True)
    x, y = ir.Var("x", "int64"), ir.Var("y", "int64")
    record = tirx.call_packed(
        "tilelang.testing.native_host_arith", prim.Div(x, y), prim.Mod(x, y), prim.FloorDiv(x, y), prim.FloorMod(x, y), ty="void"
    )
    function = host_function([x, y], [ir.Evaluate(record)])
    call, _ = compile_host(function)
    call({}, left, right)
    quotient = abs(left) // abs(right) * (-1 if (left < 0) != (right < 0) else 1)
    assert seen == [(quotient, left - quotient * right, left // right, left % right)]


def test_casts_are_not_erased_by_python_host_generation():
    seen = []
    tvm_ffi.register_global_func("tilelang.testing.native_host_cast", lambda value: seen.append(value), override=True)
    value = ir.Var("x", "uint32")
    narrowed = prim.Cast("uint32", prim.Cast("uint8", value))
    function = host_function([value], [ir.Evaluate(tirx.call_packed("tilelang.testing.native_host_cast", narrowed, ty="void"))])
    call, _ = compile_host(function)
    call({}, 256)
    call({}, 511)
    assert seen == [0, 255]


def test_tensormap_storage_is_aligned_and_lives_through_host_call():
    with HostResources(0, 0) as resources:
        first = resources.allocate("tensormap", 1)
        second = resources.allocate("tensormap", 2)
        assert first.value % 64 == 0 and second.value % 64 == 0
        ctypes.memset(first, 0x5A, 128)
        ctypes.memset(second, 0xA5, 256)
        assert ctypes.string_at(first, 128) == bytes([0x5A]) * 128
        assert ctypes.string_at(second, 256) == bytes([0xA5]) * 256


def test_assertion_keeps_exception_kind_and_message():
    value = ir.Var("n", "int32")
    function = host_function([value], [ir.AssertStmt(value > 0, ir.StringImm("ValueError"), [ir.StringImm("positive n required")])])
    call, _ = compile_host(function)
    with pytest.raises(ValueError, match="positive n required"):
        call({}, 0)


def test_missing_native_graph_is_rejected():
    function = host_function([], [])
    with pytest.raises(ValueError, match="retained by MakePackedAPI"):
        NativePythonHost(function, ir.IRModule({"main": function}), ir.IRModule({}))


@pytest.mark.parametrize("roundtrip", [False, True])
def test_lexical_regions_keep_same_name_definitions_and_effect_order(roundtrip):
    seen = []
    tvm_ffi.register_global_func("tilelang.testing.lexical_host_record", lambda value: seen.append(value), override=True)
    outer, first, second = (ir.Var("same", "int32") for _ in range(3))

    def record(value):
        return ir.Evaluate(tirx.call_packed("tilelang.testing.lexical_host_record", value, ty="void"))

    body = [
        ir.RegionStmt(
            "tl.lexical_scope",
            [],
            [],
            ir.DictAttrs({}),
            [ir.Bind(first, outer + 1), record(first)],
        ),
        ir.RegionStmt(
            "tl.lexical_scope",
            [],
            [],
            ir.DictAttrs({}),
            [ir.Bind(second, outer + 7), record(second)],
        ),
        record(outer),
    ]
    function = host_function([outer], body)
    module = retained_host(function)
    if roundtrip:
        function, module = ir.load_json(ir.save_json(function)), ir.load_json(ir.save_json(module))
    call, _ = compile_host(function, module)
    assert call({}, 4) == 0
    assert seen == [5, 11, 4]


def test_lexical_scope_preserves_loop_control_and_exception_order():
    seen = []
    tvm_ffi.register_global_func("tilelang.testing.lexical_loop_record", lambda value: seen.append(value), override=True)
    limit, index = ir.Var("limit", "int32"), ir.Var("index", "int32")
    body = ir.RegionStmt(
        "tl.lexical_scope",
        [],
        [],
        ir.DictAttrs({}),
        [
            ir.If(index == 1, [ir.Continue()], []),
            ir.AssertStmt(index < limit, ir.StringImm("ValueError"), [ir.StringImm("lexical-limit")]),
            ir.If(index == 4, [ir.Break()], []),
            ir.Evaluate(tirx.call_packed("tilelang.testing.lexical_loop_record", index, ty="void")),
        ],
    )
    function = host_function([limit], [ir.For(index, 0, 6, 1, ir.ForKind.DEFAULT, body)])
    call, _ = compile_host(function)
    assert call({}, 5) == 0
    assert seen == [0, 2, 3]
    seen.clear()
    with pytest.raises(ValueError, match="^lexical-limit$"):
        call({}, 3)
    assert seen == [0, 2]


@pytest.mark.parametrize("kind", ["other_region", "allocation_scope", "attribute_scope"])
def test_python_host_does_not_erase_unknown_or_allocation_region_semantics(kind):
    if kind == "other_region":
        body = ir.RegionStmt("tirx.compute_scope", [ir.StringImm("test")], [], ir.DictAttrs({}), [])
        message = "Unsupported region in native Python host"
    else:
        body = ir.RegionStmt("tl.lexical_scope", [], [], ir.DictAttrs({"explicit_alloc_scope": kind == "allocation_scope"}), [])
        message = "lexical scope with attributes requires explicit Python host lowering"
    with pytest.raises(TypeError, match=message):
        compile_host(host_function([], [body]))


def test_lexical_return_exits_the_existing_host_function():
    seen = []
    name = "tilelang.testing.lexical_return_record"
    tvm_ffi.register_global_func(name, lambda value: seen.append(value), override=True)
    early = ir.Var("early", "int32")
    region = ir.RegionStmt(
        "tl.lexical_scope",
        [],
        [],
        ir.DictAttrs({}),
        [
            ir.If(early > 0, [ir.Return(ir.Tuple([]))], []),
            ir.Evaluate(tirx.call_packed(name, 23, ty="void")),
        ],
    )
    function = host_function([early], [region, ir.Evaluate(tirx.call_packed(name, 31, ty="void"))])
    call, _ = compile_host(function)
    assert call({}, 1) == 0
    assert seen == []
    assert call({}, 0) == 0
    assert seen == [23, 31]


def test_lexical_local_cannot_escape_through_python_function_scope():
    local = ir.Var("local", "int32")
    region = ir.RegionStmt("tl.lexical_scope", [], [], ir.DictAttrs({}), [ir.Bind(local, ir.const(7))])
    function = host_function([], [region, ir.Evaluate(local)])
    with pytest.raises(InternalError, match="definition is no longer in-scope"):
        compile_host(function)
