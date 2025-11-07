from tilelang import tvm
from tvm import ir
import torch
import ctypes
from typing import TYPE_CHECKING, Union
from tvm import tir
import tvm.script.ir_builder.tir._ffi_api as tb_ffi

dtype = tvm.DataType
# Python 3.9 compatibility: avoid PEP 604 unions at runtime
AnyDType = Union[ir.Type, str, type, torch.dtype, dtype]

# Base dtype conversion list
_dtype_cvt_base = [
    (None, 'handle', ctypes.c_long, 'long', None),  # use long to repr void*
    (bool, 'bool', ctypes.c_bool, 'bool', 'Boolean'),
    (int, 'int32', ctypes.c_int32, 'int', 'Int32'),
    (float, 'float32', ctypes.c_float, 'float', 'Float32'),
    (torch.short, 'int16', ctypes.c_int16, 'short', 'Int16'),
    (torch.int, 'int32', ctypes.c_int32, 'int', 'Int32'),
    (torch.long, 'int64', ctypes.c_int64, 'long long', 'Int64'),
    (torch.half, 'float16', None, None, 'Float16'),
    (torch.float, 'float32', ctypes.c_float, 'float', 'Float32'),
    (torch.double, 'float64', ctypes.c_double, 'double', 'Float64'),

    #   (pytype,                'tvm dtype str',    'ctypes dtype',     'cffi dtype')
    (torch.bool, 'bool', ctypes.c_bool, 'bool', 'Boolean'),
    (torch.int8, 'int8', ctypes.c_int8, 'char', 'Int8'),
    (torch.int16, 'int16', ctypes.c_int16, 'short', 'Int16'),
    (torch.int32, 'int32', ctypes.c_int32, 'int', 'Int32'),
    (torch.int64, 'int64', ctypes.c_int64, 'long long', 'Int64'),
    (torch.uint8, 'uint8', ctypes.c_uint8, 'unsigned char', 'UInt8'),
    (torch.uint16, 'uint16', ctypes.c_uint16, 'unsigned short', 'UInt16'),
    (torch.uint32, 'uint32', ctypes.c_uint32, 'unsigned int', 'UInt32'),
    (torch.uint64, 'uint64', ctypes.c_uint64, 'unsigned long long', 'UInt64'),
    (torch.float16, 'float16', None, None, 'Float16'),
    (torch.float32, 'float32', ctypes.c_float, 'float', 'Float32'),
    (torch.float64, 'float64', ctypes.c_double, 'double', 'Float64'),
    (None, 'float8_e4m3', None, None, 'Float8E4M3'),
    (torch.bfloat16, 'bfloat16', None, None, 'BFloat16'),
]

# Dynamically add fp8-related types if they exist in torch
_fp8_dtype_mappings = [
    ('float8_e4m3fn', 'Float8E4M3FN'),
    ('float8_e4m3fnuz', 'Float8E4M3FNUZ'),
    ('float8_e5m2', 'Float8E5M2'),
    ('float8_e5m2fnuz', 'Float8E5M2FNUZ'),
    ('float8_e8m0fnu', 'Float8E8M0FNU'),
]

_dtype_cvt = list(_dtype_cvt_base)
for torch_attr_name, tvm_name in _fp8_dtype_mappings:
    if hasattr(torch, torch_attr_name):
        torch_dtype = getattr(torch, torch_attr_name)
        _dtype_cvt.append((torch_dtype, torch_attr_name, None, None, tvm_name))


def _create_type_mapper(sidx, didx, smapper=lambda x: x, dmapper=lambda x: x):
    return {
        smapper(item[sidx]): dmapper(item[didx])
        for item in _dtype_cvt
        if item[didx] is not None and item[sidx] is not None
    }


_dtype_py2tvmstr = _create_type_mapper(0, 1)
_dtype_tvmstr2fficall = _create_type_mapper(1, 4, dmapper=lambda x: getattr(tb_ffi, x))
_dtype_tvm2py = _create_type_mapper(1, 0, lambda x: dtype(x))
_dtype_tvm2ctype = _create_type_mapper(1, 2, lambda x: dtype(x))
_dtype_tvm2cffi = _create_type_mapper(1, 3, lambda x: dtype(x))


def __dtype_eq__(self: dtype, other: AnyDType):
    if isinstance(other, str):
        return str.__eq__(self, other)
    if other in _dtype_py2tvmstr:
        return str.__eq__(self, _dtype_py2tvmstr[other])
    return NotImplemented


def __dtype_ne__(self: dtype, other: AnyDType):
    if isinstance(other, str):
        return str.__ne__(self, other)
    if other in _dtype_py2tvmstr:
        return str.__ne__(self, _dtype_py2tvmstr[other])
    return NotImplemented


def __dtype_call__(self: dtype, expr=None, is_size_var: bool = False) -> tir.Var:
    if self in _dtype_tvmstr2fficall:
        return _dtype_tvmstr2fficall[self](expr, is_size_var)
    # try to construct the ffi call
    if self.startswith('uint'):
        val = 'UInt' + self[4:]
    elif self.startswith('int'):
        val = 'Int' + self[3:]
    elif self.startswith('float'):
        val = 'Float' + self[5:]
    elif self.startswith('bfloat'):
        val = 'BFloat' + self[6:]
    else:
        raise TypeError(f'Invalid type {self}')
    if '_' in val:
        first, second = val.split('_', maxsplit=1)
        val = first + second.upper()
    call = getattr(tb_ffi, val, None)
    if call is None:
        raise TypeError(f"Convert to datatype `{self}` is not supported by tvm\n"
                        f"calling failed on `tvm.script.ir_builder.tir._ffi_api.{val}`")
    return call(expr, is_size_var)


__orig_dtype_new = dtype.__new__


def __dtype_new__(cls, value: AnyDType) -> dtype:
    if isinstance(value, str):
        return __orig_dtype_new(cls, value)
    elif value in _dtype_py2tvmstr:
        return __orig_dtype_new(cls, _dtype_py2tvmstr[value])
    else:
        expected = set(list(_dtype_py2tvmstr.keys()) + list(_dtype_tvmstr2fficall.values()))
        raise TypeError(f"Invalid DataType {value}({type(value)}), expect one of {expected}")


dtype.__eq__ = __dtype_eq__
dtype.__req__ = __dtype_eq__
dtype.__ne__ = __dtype_ne__
dtype.__rne__ = __dtype_ne__
dtype.__call__ = __dtype_call__
dtype.__new__ = __dtype_new__


def get_tvm_dtype(value: AnyDType) -> dtype:
    if isinstance(value, (dtype, ir.Type)):
        return value
    return dtype(value)


if TYPE_CHECKING:

    # yapf: disable
    class bool(dtype): ...
    class short(dtype): ...
    class int(dtype): ...
    class long(dtype): ...
    class half(dtype): ...
    class float(dtype): ...
    class double(dtype): ...
    class int8(dtype): ...
    class int16(dtype): ...
    class int32(dtype): ...
    class int64(dtype): ...
    class int8x4(dtype): ...
    class int16x4(dtype): ...
    class int32x4(dtype): ...
    class int64x4(dtype): ...
    class int8x8(dtype): ...
    class int16x8(dtype): ...
    class int32x8(dtype): ...
    class int64x8(dtype): ...
    class int8x16(dtype): ...
    class int16x16(dtype): ...
    class int32x16(dtype): ...
    class int64x16(dtype): ...
    class int8x32(dtype): ...
    class int16x32(dtype): ...
    class int32x32(dtype): ...
    class int64x32(dtype): ...
    class int8x64(dtype): ...
    class int16x64(dtype): ...
    class int32x64(dtype): ...
    class int64x64(dtype): ...
    class uint8(dtype): ...
    class uint16(dtype): ...
    class uint32(dtype): ...
    class uint64(dtype): ...
    class uint8x4(dtype): ...
    class uint16x4(dtype): ...
    class uint32x4(dtype): ...
    class uint64x4(dtype): ...
    class uint8x8(dtype): ...
    class uint16x8(dtype): ...
    class uint32x8(dtype): ...
    class uint64x8(dtype): ...
    class uint8x16(dtype): ...
    class uint16x16(dtype): ...
    class uint32x16(dtype): ...
    class uint64x16(dtype): ...
    class uint8x32(dtype): ...
    class uint16x32(dtype): ...
    class uint32x32(dtype): ...
    class uint64x32(dtype): ...
    class uint8x64(dtype): ...
    class uint16x64(dtype): ...
    class uint32x64(dtype): ...
    class uint64x64(dtype): ...
    class float16(dtype): ...
    class float32(dtype): ...
    class float64(dtype): ...
    class float16x2(dtype): ...
    class float32x2(dtype): ...
    class float64x2(dtype): ...
    class float16x4(dtype): ...
    class float32x4(dtype): ...
    class float64x4(dtype): ...
    class float16x8(dtype): ...
    class float32x8(dtype): ...
    class float64x8(dtype): ...
    class float16x16(dtype): ...
    class float32x16(dtype): ...
    class float64x16(dtype): ...
    class float16x32(dtype): ...
    class float32x32(dtype): ...
    class float64x32(dtype): ...
    class float16x64(dtype): ...
    class float32x64(dtype): ...
    class float64x64(dtype): ...
    class float8_e3m4(dtype): ...
    class float8_e3m4x2(dtype): ...
    class float8_e3m4x4(dtype): ...
    class float8_e3m4x8(dtype): ...
    class float8_e3m4x16(dtype): ...
    class float8_e3m4x32(dtype): ...
    class float8_e3m4x64(dtype): ...
    class float8_e4m3(dtype): ...
    class float8_e4m3x2(dtype): ...
    class float8_e4m3x4(dtype): ...
    class float8_e4m3x8(dtype): ...
    class float8_e4m3x16(dtype): ...
    class float8_e4m3x32(dtype): ...
    class float8_e4m3x64(dtype): ...
    class float8_e4m3b11fnuz(dtype): ...
    class float8_e4m3b11fnuzx2(dtype): ...
    class float8_e4m3b11fnuzx4(dtype): ...
    class float8_e4m3b11fnuzx8(dtype): ...
    class float8_e4m3b11fnuzx16(dtype): ...
    class float8_e4m3b11fnuzx32(dtype): ...
    class float8_e4m3b11fnuzx64(dtype): ...
    class float8_e4m3fn(dtype): ...
    class float8_e4m3fnx2(dtype): ...
    class float8_e4m3fnx4(dtype): ...
    class float8_e4m3fnx8(dtype): ...
    class float8_e4m3fnx16(dtype): ...
    class float8_e4m3fnx32(dtype): ...
    class float8_e4m3fnx64(dtype): ...
    class float8_e4m3fnuz(dtype): ...
    class float8_e4m3fnuzx2(dtype): ...
    class float8_e4m3fnuzx4(dtype): ...
    class float8_e4m3fnuzx8(dtype): ...
    class float8_e4m3fnuzx16(dtype): ...
    class float8_e4m3fnuzx32(dtype): ...
    class float8_e4m3fnuzx64(dtype): ...
    class float8_e5m2(dtype): ...
    class float8_e5m2x2(dtype): ...
    class float8_e5m2x4(dtype): ...
    class float8_e5m2x8(dtype): ...
    class float8_e5m2x16(dtype): ...
    class float8_e5m2x32(dtype): ...
    class float8_e5m2x64(dtype): ...
    class float8_e5m2fnuz(dtype): ...
    class float8_e5m2fnuzx2(dtype): ...
    class float8_e5m2fnuzx4(dtype): ...
    class float8_e5m2fnuzx8(dtype): ...
    class float8_e5m2fnuzx16(dtype): ...
    class float8_e5m2fnuzx32(dtype): ...
    class float8_e5m2fnuzx64(dtype): ...
    class float8_e8m0fnu(dtype): ...
    class float8_e8m0fnux2(dtype): ...
    class float8_e8m0fnux4(dtype): ...
    class float8_e8m0fnux8(dtype): ...
    class float8_e8m0fnux16(dtype): ...
    class float8_e8m0fnux32(dtype): ...
    class float8_e8m0fnux64(dtype): ...
    class float6_e2m3fn(dtype): ...
    class float6_e2m3fnx2(dtype): ...
    class float6_e2m3fnx4(dtype): ...
    class float6_e2m3fnx8(dtype): ...
    class float6_e2m3fnx16(dtype): ...
    class float6_e2m3fnx32(dtype): ...
    class float6_e2m3fnx64(dtype): ...
    class float6_e3m2fn(dtype): ...
    class float6_e3m2fnx2(dtype): ...
    class float6_e3m2fnx4(dtype): ...
    class float6_e3m2fnx8(dtype): ...
    class float6_e3m2fnx16(dtype): ...
    class float6_e3m2fnx32(dtype): ...
    class float6_e3m2fnx64(dtype): ...
    class float4_e2m1fn(dtype): ...
    class float4_e2m1fnx2(dtype): ...
    class float4_e2m1fnx4(dtype): ...
    class float4_e2m1fnx8(dtype): ...
    class float4_e2m1fnx16(dtype): ...
    class float4_e2m1fnx32(dtype): ...
    class float4_e2m1fnx64(dtype): ...
    class bfloat16(dtype): ...
    # yapf: enable

else:
    bool = dtype('bool')
    short = dtype('int16')
    int = dtype('int32')
    long = dtype('int64')
    half = dtype('float16')
    float = dtype('float32')
    double = dtype('float64')
    int8 = dtype('int8')
    int16 = dtype('int16')
    int32 = dtype('int32')
    int64 = dtype('int64')
    int8x4 = dtype('int8x4')
    int16x4 = dtype('int16x4')
    int32x4 = dtype('int32x4')
    int64x4 = dtype('int64x4')
    int8x8 = dtype('int8x8')
    int16x8 = dtype('int16x8')
    int32x8 = dtype('int32x8')
    int64x8 = dtype('int64x8')
    int8x16 = dtype('int8x16')
    int16x16 = dtype('int16x16')
    int32x16 = dtype('int32x16')
    int64x16 = dtype('int64x16')
    int8x32 = dtype('int8x32')
    int16x32 = dtype('int16x32')
    int32x32 = dtype('int32x32')
    int64x32 = dtype('int64x32')
    int8x64 = dtype('int8x64')
    int16x64 = dtype('int16x64')
    int32x64 = dtype('int32x64')
    int64x64 = dtype('int64x64')
    uint8 = dtype('uint8')
    uint16 = dtype('uint16')
    uint32 = dtype('uint32')
    uint64 = dtype('uint64')
    uint8x4 = dtype('uint8x4')
    uint16x4 = dtype('uint16x4')
    uint32x4 = dtype('uint32x4')
    uint64x4 = dtype('uint64x4')
    uint8x8 = dtype('uint8x8')
    uint16x8 = dtype('uint16x8')
    uint32x8 = dtype('uint32x8')
    uint64x8 = dtype('uint64x8')
    uint8x16 = dtype('uint8x16')
    uint16x16 = dtype('uint16x16')
    uint32x16 = dtype('uint32x16')
    uint64x16 = dtype('uint64x16')
    uint8x32 = dtype('uint8x32')
    uint16x32 = dtype('uint16x32')
    uint32x32 = dtype('uint32x32')
    uint64x32 = dtype('uint64x32')
    uint8x64 = dtype('uint8x64')
    uint16x64 = dtype('uint16x64')
    uint32x64 = dtype('uint32x64')
    uint64x64 = dtype('uint64x64')
    float16 = dtype('float16')
    float32 = dtype('float32')
    float64 = dtype('float64')
    float16x2 = dtype('float16x2')
    float32x2 = dtype('float32x2')
    float64x2 = dtype('float64x2')
    float16x4 = dtype('float16x4')
    float32x4 = dtype('float32x4')
    float64x4 = dtype('float64x4')
    float16x8 = dtype('float16x8')
    float32x8 = dtype('float32x8')
    float64x8 = dtype('float64x8')
    float16x16 = dtype('float16x16')
    float32x16 = dtype('float32x16')
    float64x16 = dtype('float64x16')
    float16x32 = dtype('float16x32')
    float32x32 = dtype('float32x32')
    float64x32 = dtype('float64x32')
    float16x64 = dtype('float16x64')
    float32x64 = dtype('float32x64')
    float64x64 = dtype('float64x64')
    float8_e3m4 = dtype('float8_e3m4')
    float8_e3m4x2 = dtype('float8_e3m4x2')
    float8_e3m4x4 = dtype('float8_e3m4x4')
    float8_e3m4x8 = dtype('float8_e3m4x8')
    float8_e3m4x16 = dtype('float8_e3m4x16')
    float8_e3m4x32 = dtype('float8_e3m4x32')
    float8_e3m4x64 = dtype('float8_e3m4x64')
    float8_e4m3 = dtype('float8_e4m3')
    float8_e4m3x2 = dtype('float8_e4m3x2')
    float8_e4m3x4 = dtype('float8_e4m3x4')
    float8_e4m3x8 = dtype('float8_e4m3x8')
    float8_e4m3x16 = dtype('float8_e4m3x16')
    float8_e4m3x32 = dtype('float8_e4m3x32')
    float8_e4m3x64 = dtype('float8_e4m3x64')
    float8_e4m3b11fnuz = dtype('float8_e4m3b11fnuz')
    float8_e4m3b11fnuzx2 = dtype('float8_e4m3b11fnuzx2')
    float8_e4m3b11fnuzx4 = dtype('float8_e4m3b11fnuzx4')
    float8_e4m3b11fnuzx8 = dtype('float8_e4m3b11fnuzx8')
    float8_e4m3b11fnuzx16 = dtype('float8_e4m3b11fnuzx16')
    float8_e4m3b11fnuzx32 = dtype('float8_e4m3b11fnuzx32')
    float8_e4m3b11fnuzx64 = dtype('float8_e4m3b11fnuzx64')
    float8_e4m3fn = dtype('float8_e4m3fn')
    float8_e4m3fnx2 = dtype('float8_e4m3fnx2')
    float8_e4m3fnx4 = dtype('float8_e4m3fnx4')
    float8_e4m3fnx8 = dtype('float8_e4m3fnx8')
    float8_e4m3fnx16 = dtype('float8_e4m3fnx16')
    float8_e4m3fnx32 = dtype('float8_e4m3fnx32')
    float8_e4m3fnx64 = dtype('float8_e4m3fnx64')
    float8_e4m3fnuz = dtype('float8_e4m3fnuz')
    float8_e4m3fnuzx2 = dtype('float8_e4m3fnuzx2')
    float8_e4m3fnuzx4 = dtype('float8_e4m3fnuzx4')
    float8_e4m3fnuzx8 = dtype('float8_e4m3fnuzx8')
    float8_e4m3fnuzx16 = dtype('float8_e4m3fnuzx16')
    float8_e4m3fnuzx32 = dtype('float8_e4m3fnuzx32')
    float8_e4m3fnuzx64 = dtype('float8_e4m3fnuzx64')
    float8_e5m2 = dtype('float8_e5m2')
    float8_e5m2x2 = dtype('float8_e5m2x2')
    float8_e5m2x4 = dtype('float8_e5m2x4')
    float8_e5m2x8 = dtype('float8_e5m2x8')
    float8_e5m2x16 = dtype('float8_e5m2x16')
    float8_e5m2x32 = dtype('float8_e5m2x32')
    float8_e5m2x64 = dtype('float8_e5m2x64')
    float8_e5m2fnuz = dtype('float8_e5m2fnuz')
    float8_e5m2fnuzx2 = dtype('float8_e5m2fnuzx2')
    float8_e5m2fnuzx4 = dtype('float8_e5m2fnuzx4')
    float8_e5m2fnuzx8 = dtype('float8_e5m2fnuzx8')
    float8_e5m2fnuzx16 = dtype('float8_e5m2fnuzx16')
    float8_e5m2fnuzx32 = dtype('float8_e5m2fnuzx32')
    float8_e5m2fnuzx64 = dtype('float8_e5m2fnuzx64')
    float8_e8m0fnu = dtype('float8_e8m0fnu')
    float8_e8m0fnux2 = dtype('float8_e8m0fnux2')
    float8_e8m0fnux4 = dtype('float8_e8m0fnux4')
    float8_e8m0fnux8 = dtype('float8_e8m0fnux8')
    float8_e8m0fnux16 = dtype('float8_e8m0fnux16')
    float8_e8m0fnux32 = dtype('float8_e8m0fnux32')
    float8_e8m0fnux64 = dtype('float8_e8m0fnux64')
    float6_e2m3fn = dtype('float6_e2m3fn')
    float6_e2m3fnx2 = dtype('float6_e2m3fnx2')
    float6_e2m3fnx4 = dtype('float6_e2m3fnx4')
    float6_e2m3fnx8 = dtype('float6_e2m3fnx8')
    float6_e2m3fnx16 = dtype('float6_e2m3fnx16')
    float6_e2m3fnx32 = dtype('float6_e2m3fnx32')
    float6_e2m3fnx64 = dtype('float6_e2m3fnx64')
    float6_e3m2fn = dtype('float6_e3m2fn')
    float6_e3m2fnx2 = dtype('float6_e3m2fnx2')
    float6_e3m2fnx4 = dtype('float6_e3m2fnx4')
    float6_e3m2fnx8 = dtype('float6_e3m2fnx8')
    float6_e3m2fnx16 = dtype('float6_e3m2fnx16')
    float6_e3m2fnx32 = dtype('float6_e3m2fnx32')
    float6_e3m2fnx64 = dtype('float6_e3m2fnx64')
    float4_e2m1fn = dtype('float4_e2m1fn')
    float4_e2m1fnx2 = dtype('float4_e2m1fnx2')
    float4_e2m1fnx4 = dtype('float4_e2m1fnx4')
    float4_e2m1fnx8 = dtype('float4_e2m1fnx8')
    float4_e2m1fnx16 = dtype('float4_e2m1fnx16')
    float4_e2m1fnx32 = dtype('float4_e2m1fnx32')
    float4_e2m1fnx64 = dtype('float4_e2m1fnx64')
    bfloat16 = dtype('bfloat16')

_all_dtypes = {
    'bool',
    'short',
    'int',
    'long',
    'half',
    'float',
    'double',
    'int8',
    'int16',
    'int32',
    'int64',
    'int8x4',
    'int16x4',
    'int32x4',
    'int64x4',
    'int8x8',
    'int16x8',
    'int32x8',
    'int64x8',
    'int8x16',
    'int16x16',
    'int32x16',
    'int64x16',
    'int8x32',
    'int16x32',
    'int32x32',
    'int64x32',
    'int8x64',
    'int16x64',
    'int32x64',
    'int64x64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'uint8x4',
    'uint16x4',
    'uint32x4',
    'uint64x4',
    'uint8x8',
    'uint16x8',
    'uint32x8',
    'uint64x8',
    'uint8x16',
    'uint16x16',
    'uint32x16',
    'uint64x16',
    'uint8x32',
    'uint16x32',
    'uint32x32',
    'uint64x32',
    'uint8x64',
    'uint16x64',
    'uint32x64',
    'uint64x64',
    'float16',
    'float32',
    'float64',
    'float16x2',
    'float32x2',
    'float64x2',
    'float16x4',
    'float32x4',
    'float64x4',
    'float16x8',
    'float32x8',
    'float64x8',
    'float16x16',
    'float32x16',
    'float64x16',
    'float16x32',
    'float32x32',
    'float64x32',
    'float16x64',
    'float32x64',
    'float64x64',
    'float8_e3m4',
    'float8_e3m4x2',
    'float8_e3m4x4',
    'float8_e3m4x8',
    'float8_e3m4x16',
    'float8_e3m4x32',
    'float8_e3m4x64',
    'float8_e4m3',
    'float8_e4m3x2',
    'float8_e4m3x4',
    'float8_e4m3x8',
    'float8_e4m3x16',
    'float8_e4m3x32',
    'float8_e4m3x64',
    'float8_e4m3b11fnuz',
    'float8_e4m3b11fnuzx2',
    'float8_e4m3b11fnuzx4',
    'float8_e4m3b11fnuzx8',
    'float8_e4m3b11fnuzx16',
    'float8_e4m3b11fnuzx32',
    'float8_e4m3b11fnuzx64',
    'float8_e4m3fn',
    'float8_e4m3fnx2',
    'float8_e4m3fnx4',
    'float8_e4m3fnx8',
    'float8_e4m3fnx16',
    'float8_e4m3fnx32',
    'float8_e4m3fnx64',
    'float8_e4m3fnuz',
    'float8_e4m3fnuzx2',
    'float8_e4m3fnuzx4',
    'float8_e4m3fnuzx8',
    'float8_e4m3fnuzx16',
    'float8_e4m3fnuzx32',
    'float8_e4m3fnuzx64',
    'float8_e5m2',
    'float8_e5m2x2',
    'float8_e5m2x4',
    'float8_e5m2x8',
    'float8_e5m2x16',
    'float8_e5m2x32',
    'float8_e5m2x64',
    'float8_e5m2fnuz',
    'float8_e5m2fnuzx2',
    'float8_e5m2fnuzx4',
    'float8_e5m2fnuzx8',
    'float8_e5m2fnuzx16',
    'float8_e5m2fnuzx32',
    'float8_e5m2fnuzx64',
    'float8_e8m0fnu',
    'float8_e8m0fnux2',
    'float8_e8m0fnux4',
    'float8_e8m0fnux8',
    'float8_e8m0fnux16',
    'float8_e8m0fnux32',
    'float8_e8m0fnux64',
    'float6_e2m3fn',
    'float6_e2m3fnx2',
    'float6_e2m3fnx4',
    'float6_e2m3fnx8',
    'float6_e2m3fnx16',
    'float6_e2m3fnx32',
    'float6_e2m3fnx64',
    'float6_e3m2fn',
    'float6_e3m2fnx2',
    'float6_e3m2fnx4',
    'float6_e3m2fnx8',
    'float6_e3m2fnx16',
    'float6_e3m2fnx32',
    'float6_e3m2fnx64',
    'float4_e2m1fn',
    'float4_e2m1fnx2',
    'float4_e2m1fnx4',
    'float4_e2m1fnx8',
    'float4_e2m1fnx16',
    'float4_e2m1fnx32',
    'float4_e2m1fnx64',
    'bfloat16',
}

__all__ = list(_all_dtypes) + [
    'dtype',
    'AnyDType',
    'get_tvm_dtype',
]
