from tilelang import tvm
from tvm import ir
import torch
import ctypes
from typing import Any

AnyDType = ir.Type | str | type | torch.dtype | tvm.DataType

_dtype_torch2tvm = {
    # special types should placed in the first
    float: "float32",
    int: "int32",
    torch.long: "int64",
    torch.half: "half",
    # other dtypes
    torch.bool: "bool",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e4m3fnuz: "float8_e4m3fnuz",
    torch.float8_e5m2: "float8_e5m2",
    torch.float8_e5m2fnuz: "float8_e5m2fnuz",
    torch.float8_e8m0fnu: "float8_e8m0fnu",
}

_dtype_tvm2torch = {tvm.DataType(v): k for k, v in _dtype_torch2tvm.items()}

_dtype_tvm2ctype = {
    tvm.DataType("bool"): ctypes.c_bool,
    tvm.DataType("int8"): ctypes.c_int8,
    tvm.DataType("int16"): ctypes.c_int16,
    tvm.DataType("int32"): ctypes.c_int32,
    tvm.DataType("int64"): ctypes.c_int64,
    tvm.DataType("uint8"): ctypes.c_uint8,
    tvm.DataType("uint16"): ctypes.c_uint16,
    tvm.DataType("uint32"): ctypes.c_uint32,
    tvm.DataType("uint64"): ctypes.c_uint64,
    # tvm.DataType("float16"): ctypes.c_uint16,
    # tvm.DataType("bfloat16"): ctypes.c_uint16,
    tvm.DataType("float32"): ctypes.c_float,
    tvm.DataType("float64"): ctypes.c_double,
    # tvm.DataType("float8_e4m3fn"): ctypes.c_uint8,
    # tvm.DataType("float8_e4m3fnuz"): ctypes.c_uint8,
    # tvm.DataType("float8_e5m2"): ctypes.c_uint8,
    # tvm.DataType("float8_e5m2fnuz"): ctypes.c_uint8,
    # tvm.DataType("float8_e8m0fnu"): ctypes.c_uint8,
    tvm.DataType("handle"): ctypes.c_void_p,
}

_dtype_tvm2cffi = {
    tvm.DataType("bool"): "bool",
    tvm.DataType("int8"): "char",
    tvm.DataType("int16"): "short",
    tvm.DataType("int32"): "int",
    tvm.DataType("int64"): "long long",
    tvm.DataType("uint8"): "unsigned char",
    tvm.DataType("uint16"): "unsigned short",
    tvm.DataType("uint32"): "unsigned int",
    tvm.DataType("uint64"): "unsigned long long",
    tvm.DataType("float32"): "float",
    tvm.DataType("float64"): "double",
    # tvm.DataType("float16"): 'uint16_t',
    # tvm.DataType("bfloat16"): 'uint16_t',
    # tvm.DataType("float8_e4m3fn"): 'uint8_t',
    # tvm.DataType("float8_e4m3fnuz"): 'uint8_t',
    # tvm.DataType("float8_e5m2"): ctypes.c_uint8,
    # tvm.DataType("float8_e5m2fnuz"): ctypes.c_uint8,
    # tvm.DataType("float8_e8m0fnu"): ctypes.c_uint8,
    tvm.DataType("handle"): "long",
}


def get_tvm_dtype(ty: AnyDType) -> tvm.DataType:
    if isinstance(ty, (ir.Type, tvm.DataType)):
        return ty
    if isinstance(ty, str):
        return tvm.DataType(ty)
    return tvm.DataType(_dtype_torch2tvm[ty])


def get_torch_dtype(ty: AnyDType) -> torch.dtype:
    if isinstance(ty, torch.dtype):
        return ty
    if isinstance(ty, str):
        ty = tvm.DataType(ty)
    return _dtype_tvm2torch[ty]


def get_ctypes_dtype(ty: AnyDType) -> Any:
    ty = get_tvm_dtype(ty)
    return _dtype_tvm2ctype[ty]


def get_cffi_dtype(ty: AnyDType) -> str:
    ty = get_tvm_dtype(ty)
    return _dtype_tvm2cffi[ty]


def get_tvm_ptr_type(
    ty: ir.Type | str | type | torch.dtype, scope: str = "global"
) -> ir.PointerType:
    ty = get_tvm_dtype(ty)
    return ir.PointerType(ir.PrimType(ty), scope)
