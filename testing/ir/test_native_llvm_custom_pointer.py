"""Host wrappers transport device custom pointers without scalar legalization."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, target, tirx

# TileLang registers these device element types when its compiler loads.
import tilelang  # noqa: F401


LLVM = target.Target({"kind": "llvm", "mcpu": "generic"})
DTYPES = ["custom[tfloat32]32", "custom[float4_e2m1_unpacked]8"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lanes", [1, 4])
def test_custom_pointer_null_check_and_address_transport(dtype, lanes):
    element = tvm_ffi.dtype(dtype).with_lanes(lanes)
    pointer = ir.Var("pointer", ir.PtrType(ir.PrimType(element)))
    # This is the host-side operation that previously requested a nonexistent
    # LLVM scalar element type, then dereferenced a null llvm::Type pointer.
    function = tirx.Function([pointer], ir.Return(tirx.isnullptr(pointer)), ir.PrimType("bool")).with_attr(
        {"global_symbol": "main", "target": LLVM}
    )
    compiled = tirx.compile(function, target=LLVM)
    assert compiled(ctypes.c_void_p()) is True
    storage = (ctypes.c_uint8 * 64)()
    assert compiled(ctypes.cast(storage, ctypes.c_void_p)) is False


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 7])
def test_custom_pointer_roundtrip_and_byte_offset_preserve_the_address(dtype, offset):
    pointer = ir.Var("pointer", ir.PtrType(dtype))
    moved = tirx.ptr_byte_offset(pointer, offset, ty=pointer.ty)
    result = tirx.reinterpret(moved, ty="uint64").astype("int64")
    function = tirx.Function([pointer], ir.Return(result), ir.PrimType("int64")).with_attr({"global_symbol": "main", "target": LLVM})
    compiled = tirx.compile(function, target=LLVM)
    storage = (ctypes.c_uint8 * 64)()
    address = ctypes.addressof(storage)
    assert compiled(ctypes.c_void_p(address)) == address + offset


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("nested", [False, True])
def test_custom_pointer_debug_information_keeps_the_custom_element_name(dtype, nested):
    location = ir.SourceLoc(ir.SourceName("custom_pointer.py"), 17, 3, 17, 21)
    pointer_type = ir.PtrType(ir.PtrType(dtype)) if nested else ir.PtrType(dtype)
    pointer = ir.Var("pointer", pointer_type, loc=location)
    function = tirx.Function(
        [pointer],
        [
            ir.Evaluate(tirx.isnullptr(pointer), loc=location),
            ir.Return(ir.const(0, "int32"), loc=location),
        ],
        ir.PrimType("int32"),
        loc=location,
    ).with_attr({"global_symbol": "custom_pointer", "target": LLVM})
    module = tvm_ffi.get_global_func("target.build.llvm")(ir.IRModule({"custom_pointer": function}), LLVM)
    source = module.inspect_source("ll")
    assert "DW_TAG_pointer_type" in source
    canonical_dtype = tvm_ffi.dtype(dtype).with_lanes(1)
    assert f'name: "{canonical_dtype}"' in source
    assert "!DILocation(line: 17, column: 3," in source


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lanes", [1, 4])
def test_custom_scalar_or_vector_requires_explicit_storage_legalization(dtype, lanes):
    element = tvm_ffi.dtype(dtype).with_lanes(lanes)
    value = ir.Var("value", ir.PrimType(element))
    function = tirx.Function([value], ir.Return(ir.const(0, "int32")), ir.PrimType("int32")).with_attr(
        {"global_symbol": "unlegalized", "target": LLVM}
    )
    with pytest.raises(TypeError, match="LLVM requires storage legalization for primitive type"):
        tvm_ffi.get_global_func("target.build.llvm")(ir.IRModule({"unlegalized": function}), LLVM)
