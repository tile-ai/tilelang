"""Packed C host calls retain their values, effects, and error propagation."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("int32", (-7, 0, 11)),
        ("uint32", (0, 17, 3_200_000_000)),
        ("bool", (False, True)),
        ("float32", (-2.5, 0.0, 3.25)),
        ("float64", (-2.5, 0.0, 3.25)),
    ],
)
@pytest.mark.parametrize("public", [False, True])
def test_helper_result_is_returned_to_its_caller(dtype, values, public):
    leaf = ir.GlobalVar("packed_result_leaf")
    x, y = ir.Var("x", dtype), ir.Var("y", dtype)
    attrs = {"global_symbol": leaf.name_hint} if public else {}
    helper = tirx.Function([y], ir.Return(y), y.ty, ir.DictAttrs(attrs))
    main = tirx.Function(
        [x],
        ir.Return(ir.Call(leaf, [x], ty=x.ty)),
        x.ty,
        ir.DictAttrs({"global_symbol": "main"}),
    )
    executable = compile_packed_function(ir.IRModule({"main": main, leaf: helper}))
    for value in values:
        result = executable(value)
        assert type(result) is type(value)
        assert result == value


def test_public_pointer_helper_preserves_the_address():
    leaf = ir.GlobalVar("packed_pointer_leaf")
    x, y = ir.Var("x", ir.PtrType("uint8")), ir.Var("y", ir.PtrType("uint8"))
    helper = tirx.Function(
        [y],
        ir.Return(tirx.ptr_byte_offset(y, 7, ty=y.ty)),
        y.ty,
        ir.DictAttrs({"global_symbol": leaf.name_hint}),
    )
    main = tirx.Function(
        [x],
        ir.Return(ir.Call(leaf, [x], ty=x.ty)),
        x.ty,
        ir.DictAttrs({"global_symbol": "main"}),
    )
    executable = compile_packed_function(ir.IRModule({"main": main, leaf: helper}))
    backing = (ctypes.c_uint8 * 32)()
    address = ctypes.addressof(backing)
    result = executable(ctypes.c_void_p(address))
    assert isinstance(result, ctypes.c_void_p)
    assert result.value == address + 7


@pytest.mark.parametrize("dtype", ["float4_e2m1fn", "bfloat16", "float32x4"])
def test_device_element_pointer_keeps_its_address_and_byte_offset(dtype):
    pointer = ir.Var("pointer", ir.PtrType(dtype))
    offset = ir.Var("offset", "int64")
    function = tirx.Function(
        [pointer, offset],
        ir.Return(tirx.ptr_byte_offset(pointer, offset, ty=pointer.ty)),
        pointer.ty,
        ir.DictAttrs({"global_symbol": "main"}),
    )
    executable = compile_packed_function(function)
    storage = (ctypes.c_uint8 * 64)()
    address = ctypes.addressof(storage) + 16
    for delta in (0, 7, -3):
        result = executable(ctypes.c_void_p(address), delta)
        assert isinstance(result, ctypes.c_void_p)
        assert result.value == address + delta


def test_explicit_registry_call_bindings_keep_results_and_propagate_errors():
    seen = []
    name = "tl.testing.packed_result_callback"

    def callback(value):
        seen.append(value)
        if value < 0:
            raise ValueError("packed-result-negative")
        return value + 5

    tvm_ffi.register_global_func(name, callback, override=True)
    x = ir.Var("x", "int32")
    first = tirx.call_packed(name, x, ty="int32")
    second = tirx.call_packed(name, x + 1, ty="int32")
    first_value = ir.Var("first_value", "int32")
    second_value = ir.Var("second_value", "int32")
    # Explicit statement order is the LowerTVMBuiltin input contract used here.
    # Sibling calls in a single expression have a separate pre-existing stack
    # reuse issue; this case does not claim to fix that lowering behavior.
    function = tirx.Function(
        [x],
        [ir.Bind(first_value, first), ir.Bind(second_value, second), ir.Return(first_value + 10 * second_value)],
        x.ty,
        ir.DictAttrs({"global_symbol": "main"}),
    )
    executable = compile_packed_function(function)
    assert executable(2) == 7 + 10 * 8
    assert seen == [2, 3]
    seen.clear()
    with pytest.raises(ValueError, match="packed-result-negative"):
        executable(-1)
    assert seen == [-1]
    seen.clear()
    assert executable(7) == 12 + 10 * 13
    assert seen == [7, 8]


def test_failing_equality_reports_original_message_without_reevaluation():
    seen = []
    name = "tl.testing.packed_local.assert_effect"

    def callback(value):
        seen.append(value)
        return value

    tvm_ffi.register_global_func(name, callback, override=True)
    x = ir.Var("x", "int32")
    condition = tirx.call_packed(name, x, ty="int32") == 0
    function = tirx.Function(
        [x],
        [
            ir.AssertStmt(condition, ir.StringImm("AssertionError"), [ir.StringImm("body-condition")]),
            ir.Return(ir.const(37, "int32")),
        ],
        ir.PrimType("int32"),
        ir.DictAttrs({"global_symbol": "main"}),
    )
    executable = compile_packed_function(function)
    with pytest.raises(AssertionError, match="^body-condition$"):
        executable(1)
    assert seen == [1]
    assert executable(0) == 37
    assert seen == [1, 0]
