"""Custom element pointers are opaque addresses in the LLVM host wrapper."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, target, tirx


LLVM = target.Target({"kind": "llvm", "mcpu": "generic"})
CUSTOM = ["custom[tfloat32]32", "custom[float4_e2m1_unpacked]8"]


def _function(parameters, body, ret_type="int32"):
    if isinstance(ret_type, str):
        ret_type = ir.PrimType(ret_type)
    return tirx.Function(parameters, body, ret_type).with_attr({"global_symbol": "main", "target": LLVM})


@pytest.mark.parametrize("dtype", CUSTOM)
@pytest.mark.parametrize("offset", [0, 7, -3])
def test_custom_pointer_cast_offset_and_return_preserve_address(dtype, offset):
    pointer_type = ir.PtrType(dtype)
    pointer = ir.Var("pointer", ir.PtrType("void"))
    typed = ir.Var("typed_pointer", pointer_type)
    result = tirx.ptr_byte_offset(typed, ir.const(offset, "int64"), ty=pointer_type)
    function = _function(
        [pointer],
        [ir.Bind(typed, tirx.reinterpret(pointer, ty=pointer_type)), ir.Return(result)],
        pointer_type,
    )
    compiled = tirx.compile(function, target=LLVM)
    storage = (ctypes.c_uint8 * 32)()
    base = ctypes.addressof(storage) + 8
    actual = compiled(ctypes.c_void_p(base))
    assert isinstance(actual, ctypes.c_void_p)
    assert actual.value == base + offset


@pytest.mark.parametrize("dtype", CUSTOM)
def test_custom_pointer_null_checks_do_not_require_a_host_scalar_type(dtype):
    pointer = ir.Var("pointer", ir.PtrType(dtype))
    function = _function([pointer], ir.Return(tirx.isnullptr(pointer)), "bool")
    compiled = tirx.compile(function, target=LLVM)
    assert compiled(ctypes.c_void_p()) is True
    storage = (ctypes.c_uint8 * 8)()
    assert compiled(ctypes.c_void_p(ctypes.addressof(storage))) is False


@pytest.mark.parametrize("dtype", CUSTOM)
def test_pointer_to_custom_pointer_has_a_valid_native_signature(dtype):
    pointer_type = ir.PtrType(ir.PtrType(dtype))
    pointer = ir.Var("pointer", pointer_type)
    compiled = tirx.compile(_function([pointer], ir.Return(pointer), pointer_type), target=LLVM)
    storage = ctypes.c_void_p(0x12345678)
    address = ctypes.c_void_p(ctypes.addressof(storage))
    assert compiled(address).value == address.value


@pytest.mark.parametrize("dtype", CUSTOM)
def test_unlowered_custom_scalar_is_rejected_with_a_diagnostic(dtype):
    value = ir.Var("value", ir.PrimType(dtype))
    function = _function([value], ir.Return(value), ir.PrimType(dtype))
    with pytest.raises(TypeError, match="storage legalization for primitive type"):
        tvm_ffi.get_global_func("target.build.llvm")(ir.IRModule.from_expr(function), LLVM)
