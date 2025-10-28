from tilelang import tvm
from tvm import ir
import torch
import ctypes
from typing import TYPE_CHECKING
from tvm import tir
import tvm.script.ir_builder.tir._ffi_api as tb_ffi


class VoidPtr:
    ...


AnyDType = ir.Type | str | type | torch.dtype | tvm.DataType

_dtype_cvt = [
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
    (torch.float8_e4m3fn, 'float8_e4m3fn', None, None, 'Float8E4M3FN'),
    (torch.float8_e4m3fnuz, 'float8_e4m3fnuz', None, None, 'Float8E4M3FNUZ'),
    (torch.float8_e5m2, 'float8_e5m2', None, None, 'Float8E5M2'),
    (torch.float8_e5m2fnuz, 'float8_e5m2fnuz', None, None, 'Float8E5M2FNUZ'),
    (torch.float8_e8m0fnu, 'float8_e8m0fnu', None, None, 'Float8E8M0FNU'),
    (torch.bfloat16, 'bfloat16', None, None, 'BFloat16'),
]


def _create_type_mapper(sidx, didx, smapper=lambda x: x, dmapper=lambda x: x):
    return {
        smapper(item[sidx]): dmapper(item[didx])
        for item in _dtype_cvt
        if item[didx] is not None and item[sidx] is not None
    }


_dtype_py2tvmstr = _create_type_mapper(0, 1)
_dtype_tvmstr2fficall = _create_type_mapper(1, 4, dmapper=lambda x: getattr(tb_ffi, x))
_dtype_tvm2py = _create_type_mapper(1, 0, lambda x: tvm.DataType(x))
_dtype_tvm2ctype = _create_type_mapper(1, 2, lambda x: tvm.DataType(x))
_dtype_tvm2cffi = _create_type_mapper(1, 3, lambda x: tvm.DataType(x))


def __dtype_eq__(self: tvm.DataType, other: AnyDType):
    if isinstance(other, str):
        return str.__eq__(self, other)
    if other in _dtype_py2tvmstr:
        return str.__eq__(self, _dtype_py2tvmstr[other])
    return NotImplemented


def __dtype_ne__(self: tvm.DataType, other: AnyDType):
    if isinstance(other, str):
        return str.__ne__(self, other)
    if other in _dtype_py2tvmstr:
        return str.__ne__(self, _dtype_py2tvmstr[other])
    return NotImplemented


def __dtype_call__(self: tvm.DataType, expr=None, is_size_var: bool = False) -> tir.Var:
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
        raise TypeError(f'Convert to datatype `{self}` is not supported by tvm, calling failed on `tvm.script.ir_builder.tir._ffi_api.{val}`')
    return call(expr, is_size_var)


def __dtype_new__(cls, value: AnyDType) -> tvm.DataType:
    if isinstance(value, str):
        val = str.__new__(cls, value)
    elif value in _dtype_py2tvmstr:
        val = str.__new__(cls, _dtype_py2tvmstr[value])
    else:
        expected = set(list(_dtype_py2tvmstr.keys()) + list(_dtype_tvmstr2fficall.values()))
        raise TypeError(f"Invalid DataType {value}({type(value)}), expect one of {expected}")
    val.__tvm_ffi_dtype__ = tvm.ffi.core.DataType(val)
    return val


tvm.DataType.__eq__ = __dtype_eq__
tvm.DataType.__req__ = __dtype_eq__
tvm.DataType.__ne__ = __dtype_ne__
tvm.DataType.__rne__ = __dtype_ne__
tvm.DataType.__call__ = __dtype_call__
tvm.DataType.__new__ = __dtype_new__


def get_tvm_dtype(value: AnyDType) -> tvm.DataType:
    if isinstance(value, (tvm.DataType, ir.Type)):
        return value
    return tvm.DataType(value)


if TYPE_CHECKING:
    class bool(tvm.DataType): ...
    class short(tvm.DataType): ...
    class int(tvm.DataType): ...
    class long(tvm.DataType): ...
    class half(tvm.DataType): ...
    class float(tvm.DataType): ...
    class double(tvm.DataType): ...
    class int8(tvm.DataType): ...
    class int16(tvm.DataType): ...
    class int32(tvm.DataType): ...
    class int64(tvm.DataType): ...
    class int8x4(tvm.DataType): ...
    class int16x4(tvm.DataType): ...
    class int32x4(tvm.DataType): ...
    class int64x4(tvm.DataType): ...
    class int8x8(tvm.DataType): ...
    class int16x8(tvm.DataType): ...
    class int32x8(tvm.DataType): ...
    class int64x8(tvm.DataType): ...
    class int8x16(tvm.DataType): ...
    class int16x16(tvm.DataType): ...
    class int32x16(tvm.DataType): ...
    class int64x16(tvm.DataType): ...
    class int8x32(tvm.DataType): ...
    class int16x32(tvm.DataType): ...
    class int32x32(tvm.DataType): ...
    class int64x32(tvm.DataType): ...
    class int8x64(tvm.DataType): ...
    class int16x64(tvm.DataType): ...
    class int32x64(tvm.DataType): ...
    class int64x64(tvm.DataType): ...
    class uint8(tvm.DataType): ...
    class uint16(tvm.DataType): ...
    class uint32(tvm.DataType): ...
    class uint64(tvm.DataType): ...
    class uint8x4(tvm.DataType): ...
    class uint16x4(tvm.DataType): ...
    class uint32x4(tvm.DataType): ...
    class uint64x4(tvm.DataType): ...
    class uint8x8(tvm.DataType): ...
    class uint16x8(tvm.DataType): ...
    class uint32x8(tvm.DataType): ...
    class uint64x8(tvm.DataType): ...
    class uint8x16(tvm.DataType): ...
    class uint16x16(tvm.DataType): ...
    class uint32x16(tvm.DataType): ...
    class uint64x16(tvm.DataType): ...
    class uint8x32(tvm.DataType): ...
    class uint16x32(tvm.DataType): ...
    class uint32x32(tvm.DataType): ...
    class uint64x32(tvm.DataType): ...
    class uint8x64(tvm.DataType): ...
    class uint16x64(tvm.DataType): ...
    class uint32x64(tvm.DataType): ...
    class uint64x64(tvm.DataType): ...
    class float16(tvm.DataType): ...
    class float32(tvm.DataType): ...
    class float64(tvm.DataType): ...
    class float16x2(tvm.DataType): ...
    class float32x2(tvm.DataType): ...
    class float64x2(tvm.DataType): ...
    class float16x4(tvm.DataType): ...
    class float32x4(tvm.DataType): ...
    class float64x4(tvm.DataType): ...
    class float16x8(tvm.DataType): ...
    class float32x8(tvm.DataType): ...
    class float64x8(tvm.DataType): ...
    class float16x16(tvm.DataType): ...
    class float32x16(tvm.DataType): ...
    class float64x16(tvm.DataType): ...
    class float16x32(tvm.DataType): ...
    class float32x32(tvm.DataType): ...
    class float64x32(tvm.DataType): ...
    class float16x64(tvm.DataType): ...
    class float32x64(tvm.DataType): ...
    class float64x64(tvm.DataType): ...
    class float8_e3m4(tvm.DataType): ...
    class float8_e3m4x2(tvm.DataType): ...
    class float8_e3m4x4(tvm.DataType): ...
    class float8_e3m4x8(tvm.DataType): ...
    class float8_e3m4x16(tvm.DataType): ...
    class float8_e3m4x32(tvm.DataType): ...
    class float8_e3m4x64(tvm.DataType): ...
    class float8_e4m3(tvm.DataType): ...
    class float8_e4m3x2(tvm.DataType): ...
    class float8_e4m3x4(tvm.DataType): ...
    class float8_e4m3x8(tvm.DataType): ...
    class float8_e4m3x16(tvm.DataType): ...
    class float8_e4m3x32(tvm.DataType): ...
    class float8_e4m3x64(tvm.DataType): ...
    class float8_e4m3b11fnuz(tvm.DataType): ...
    class float8_e4m3b11fnuzx2(tvm.DataType): ...
    class float8_e4m3b11fnuzx4(tvm.DataType): ...
    class float8_e4m3b11fnuzx8(tvm.DataType): ...
    class float8_e4m3b11fnuzx16(tvm.DataType): ...
    class float8_e4m3b11fnuzx32(tvm.DataType): ...
    class float8_e4m3b11fnuzx64(tvm.DataType): ...
    class float8_e4m3fn(tvm.DataType): ...
    class float8_e4m3fnx2(tvm.DataType): ...
    class float8_e4m3fnx4(tvm.DataType): ...
    class float8_e4m3fnx8(tvm.DataType): ...
    class float8_e4m3fnx16(tvm.DataType): ...
    class float8_e4m3fnx32(tvm.DataType): ...
    class float8_e4m3fnx64(tvm.DataType): ...
    class float8_e4m3fnuz(tvm.DataType): ...
    class float8_e4m3fnuzx2(tvm.DataType): ...
    class float8_e4m3fnuzx4(tvm.DataType): ...
    class float8_e4m3fnuzx8(tvm.DataType): ...
    class float8_e4m3fnuzx16(tvm.DataType): ...
    class float8_e4m3fnuzx32(tvm.DataType): ...
    class float8_e4m3fnuzx64(tvm.DataType): ...
    class float8_e5m2(tvm.DataType): ...
    class float8_e5m2x2(tvm.DataType): ...
    class float8_e5m2x4(tvm.DataType): ...
    class float8_e5m2x8(tvm.DataType): ...
    class float8_e5m2x16(tvm.DataType): ...
    class float8_e5m2x32(tvm.DataType): ...
    class float8_e5m2x64(tvm.DataType): ...
    class float8_e5m2fnuz(tvm.DataType): ...
    class float8_e5m2fnuzx2(tvm.DataType): ...
    class float8_e5m2fnuzx4(tvm.DataType): ...
    class float8_e5m2fnuzx8(tvm.DataType): ...
    class float8_e5m2fnuzx16(tvm.DataType): ...
    class float8_e5m2fnuzx32(tvm.DataType): ...
    class float8_e5m2fnuzx64(tvm.DataType): ...
    class float8_e8m0fnu(tvm.DataType): ...
    class float8_e8m0fnux2(tvm.DataType): ...
    class float8_e8m0fnux4(tvm.DataType): ...
    class float8_e8m0fnux8(tvm.DataType): ...
    class float8_e8m0fnux16(tvm.DataType): ...
    class float8_e8m0fnux32(tvm.DataType): ...
    class float8_e8m0fnux64(tvm.DataType): ...
    class float6_e2m3fn(tvm.DataType): ...
    class float6_e2m3fnx2(tvm.DataType): ...
    class float6_e2m3fnx4(tvm.DataType): ...
    class float6_e2m3fnx8(tvm.DataType): ...
    class float6_e2m3fnx16(tvm.DataType): ...
    class float6_e2m3fnx32(tvm.DataType): ...
    class float6_e2m3fnx64(tvm.DataType): ...
    class float6_e3m2fn(tvm.DataType): ...
    class float6_e3m2fnx2(tvm.DataType): ...
    class float6_e3m2fnx4(tvm.DataType): ...
    class float6_e3m2fnx8(tvm.DataType): ...
    class float6_e3m2fnx16(tvm.DataType): ...
    class float6_e3m2fnx32(tvm.DataType): ...
    class float6_e3m2fnx64(tvm.DataType): ...
    class float4_e2m1fn(tvm.DataType): ...
    class float4_e2m1fnx2(tvm.DataType): ...
    class float4_e2m1fnx4(tvm.DataType): ...
    class float4_e2m1fnx8(tvm.DataType): ...
    class float4_e2m1fnx16(tvm.DataType): ...
    class float4_e2m1fnx32(tvm.DataType): ...
    class float4_e2m1fnx64(tvm.DataType): ...
    class bfloat16(tvm.DataType): ...
else:
    bool = tvm.DataType('bool')
    short = tvm.DataType('int16')
    int = tvm.DataType('int32')
    long = tvm.DataType('int64')
    half = tvm.DataType('float16')
    float = tvm.DataType('float32')
    double = tvm.DataType('float64')
    int8 = tvm.DataType('int8')
    int16 = tvm.DataType('int16')
    int32 = tvm.DataType('int32')
    int64 = tvm.DataType('int64')
    int8x4 = tvm.DataType('int8x4')
    int16x4 = tvm.DataType('int16x4')
    int32x4 = tvm.DataType('int32x4')
    int64x4 = tvm.DataType('int64x4')
    int8x8 = tvm.DataType('int8x8')
    int16x8 = tvm.DataType('int16x8')
    int32x8 = tvm.DataType('int32x8')
    int64x8 = tvm.DataType('int64x8')
    int8x16 = tvm.DataType('int8x16')
    int16x16 = tvm.DataType('int16x16')
    int32x16 = tvm.DataType('int32x16')
    int64x16 = tvm.DataType('int64x16')
    int8x32 = tvm.DataType('int8x32')
    int16x32 = tvm.DataType('int16x32')
    int32x32 = tvm.DataType('int32x32')
    int64x32 = tvm.DataType('int64x32')
    int8x64 = tvm.DataType('int8x64')
    int16x64 = tvm.DataType('int16x64')
    int32x64 = tvm.DataType('int32x64')
    int64x64 = tvm.DataType('int64x64')
    uint8 = tvm.DataType('uint8')
    uint16 = tvm.DataType('uint16')
    uint32 = tvm.DataType('uint32')
    uint64 = tvm.DataType('uint64')
    uint8x4 = tvm.DataType('uint8x4')
    uint16x4 = tvm.DataType('uint16x4')
    uint32x4 = tvm.DataType('uint32x4')
    uint64x4 = tvm.DataType('uint64x4')
    uint8x8 = tvm.DataType('uint8x8')
    uint16x8 = tvm.DataType('uint16x8')
    uint32x8 = tvm.DataType('uint32x8')
    uint64x8 = tvm.DataType('uint64x8')
    uint8x16 = tvm.DataType('uint8x16')
    uint16x16 = tvm.DataType('uint16x16')
    uint32x16 = tvm.DataType('uint32x16')
    uint64x16 = tvm.DataType('uint64x16')
    uint8x32 = tvm.DataType('uint8x32')
    uint16x32 = tvm.DataType('uint16x32')
    uint32x32 = tvm.DataType('uint32x32')
    uint64x32 = tvm.DataType('uint64x32')
    uint8x64 = tvm.DataType('uint8x64')
    uint16x64 = tvm.DataType('uint16x64')
    uint32x64 = tvm.DataType('uint32x64')
    uint64x64 = tvm.DataType('uint64x64')
    float16 = tvm.DataType('float16')
    float32 = tvm.DataType('float32')
    float64 = tvm.DataType('float64')
    float16x2 = tvm.DataType('float16x2')
    float32x2 = tvm.DataType('float32x2')
    float64x2 = tvm.DataType('float64x2')
    float16x4 = tvm.DataType('float16x4')
    float32x4 = tvm.DataType('float32x4')
    float64x4 = tvm.DataType('float64x4')
    float16x8 = tvm.DataType('float16x8')
    float32x8 = tvm.DataType('float32x8')
    float64x8 = tvm.DataType('float64x8')
    float16x16 = tvm.DataType('float16x16')
    float32x16 = tvm.DataType('float32x16')
    float64x16 = tvm.DataType('float64x16')
    float16x32 = tvm.DataType('float16x32')
    float32x32 = tvm.DataType('float32x32')
    float64x32 = tvm.DataType('float64x32')
    float16x64 = tvm.DataType('float16x64')
    float32x64 = tvm.DataType('float32x64')
    float64x64 = tvm.DataType('float64x64')
    float8_e3m4 = tvm.DataType('float8_e3m4')
    float8_e3m4x2 = tvm.DataType('float8_e3m4x2')
    float8_e3m4x4 = tvm.DataType('float8_e3m4x4')
    float8_e3m4x8 = tvm.DataType('float8_e3m4x8')
    float8_e3m4x16 = tvm.DataType('float8_e3m4x16')
    float8_e3m4x32 = tvm.DataType('float8_e3m4x32')
    float8_e3m4x64 = tvm.DataType('float8_e3m4x64')
    float8_e4m3 = tvm.DataType('float8_e4m3')
    float8_e4m3x2 = tvm.DataType('float8_e4m3x2')
    float8_e4m3x4 = tvm.DataType('float8_e4m3x4')
    float8_e4m3x8 = tvm.DataType('float8_e4m3x8')
    float8_e4m3x16 = tvm.DataType('float8_e4m3x16')
    float8_e4m3x32 = tvm.DataType('float8_e4m3x32')
    float8_e4m3x64 = tvm.DataType('float8_e4m3x64')
    float8_e4m3b11fnuz = tvm.DataType('float8_e4m3b11fnuz')
    float8_e4m3b11fnuzx2 = tvm.DataType('float8_e4m3b11fnuzx2')
    float8_e4m3b11fnuzx4 = tvm.DataType('float8_e4m3b11fnuzx4')
    float8_e4m3b11fnuzx8 = tvm.DataType('float8_e4m3b11fnuzx8')
    float8_e4m3b11fnuzx16 = tvm.DataType('float8_e4m3b11fnuzx16')
    float8_e4m3b11fnuzx32 = tvm.DataType('float8_e4m3b11fnuzx32')
    float8_e4m3b11fnuzx64 = tvm.DataType('float8_e4m3b11fnuzx64')
    float8_e4m3fn = tvm.DataType('float8_e4m3fn')
    float8_e4m3fnx2 = tvm.DataType('float8_e4m3fnx2')
    float8_e4m3fnx4 = tvm.DataType('float8_e4m3fnx4')
    float8_e4m3fnx8 = tvm.DataType('float8_e4m3fnx8')
    float8_e4m3fnx16 = tvm.DataType('float8_e4m3fnx16')
    float8_e4m3fnx32 = tvm.DataType('float8_e4m3fnx32')
    float8_e4m3fnx64 = tvm.DataType('float8_e4m3fnx64')
    float8_e4m3fnuz = tvm.DataType('float8_e4m3fnuz')
    float8_e4m3fnuzx2 = tvm.DataType('float8_e4m3fnuzx2')
    float8_e4m3fnuzx4 = tvm.DataType('float8_e4m3fnuzx4')
    float8_e4m3fnuzx8 = tvm.DataType('float8_e4m3fnuzx8')
    float8_e4m3fnuzx16 = tvm.DataType('float8_e4m3fnuzx16')
    float8_e4m3fnuzx32 = tvm.DataType('float8_e4m3fnuzx32')
    float8_e4m3fnuzx64 = tvm.DataType('float8_e4m3fnuzx64')
    float8_e5m2 = tvm.DataType('float8_e5m2')
    float8_e5m2x2 = tvm.DataType('float8_e5m2x2')
    float8_e5m2x4 = tvm.DataType('float8_e5m2x4')
    float8_e5m2x8 = tvm.DataType('float8_e5m2x8')
    float8_e5m2x16 = tvm.DataType('float8_e5m2x16')
    float8_e5m2x32 = tvm.DataType('float8_e5m2x32')
    float8_e5m2x64 = tvm.DataType('float8_e5m2x64')
    float8_e5m2fnuz = tvm.DataType('float8_e5m2fnuz')
    float8_e5m2fnuzx2 = tvm.DataType('float8_e5m2fnuzx2')
    float8_e5m2fnuzx4 = tvm.DataType('float8_e5m2fnuzx4')
    float8_e5m2fnuzx8 = tvm.DataType('float8_e5m2fnuzx8')
    float8_e5m2fnuzx16 = tvm.DataType('float8_e5m2fnuzx16')
    float8_e5m2fnuzx32 = tvm.DataType('float8_e5m2fnuzx32')
    float8_e5m2fnuzx64 = tvm.DataType('float8_e5m2fnuzx64')
    float8_e8m0fnu = tvm.DataType('float8_e8m0fnu')
    float8_e8m0fnux2 = tvm.DataType('float8_e8m0fnux2')
    float8_e8m0fnux4 = tvm.DataType('float8_e8m0fnux4')
    float8_e8m0fnux8 = tvm.DataType('float8_e8m0fnux8')
    float8_e8m0fnux16 = tvm.DataType('float8_e8m0fnux16')
    float8_e8m0fnux32 = tvm.DataType('float8_e8m0fnux32')
    float8_e8m0fnux64 = tvm.DataType('float8_e8m0fnux64')
    float6_e2m3fn = tvm.DataType('float6_e2m3fn')
    float6_e2m3fnx2 = tvm.DataType('float6_e2m3fnx2')
    float6_e2m3fnx4 = tvm.DataType('float6_e2m3fnx4')
    float6_e2m3fnx8 = tvm.DataType('float6_e2m3fnx8')
    float6_e2m3fnx16 = tvm.DataType('float6_e2m3fnx16')
    float6_e2m3fnx32 = tvm.DataType('float6_e2m3fnx32')
    float6_e2m3fnx64 = tvm.DataType('float6_e2m3fnx64')
    float6_e3m2fn = tvm.DataType('float6_e3m2fn')
    float6_e3m2fnx2 = tvm.DataType('float6_e3m2fnx2')
    float6_e3m2fnx4 = tvm.DataType('float6_e3m2fnx4')
    float6_e3m2fnx8 = tvm.DataType('float6_e3m2fnx8')
    float6_e3m2fnx16 = tvm.DataType('float6_e3m2fnx16')
    float6_e3m2fnx32 = tvm.DataType('float6_e3m2fnx32')
    float6_e3m2fnx64 = tvm.DataType('float6_e3m2fnx64')
    float4_e2m1fn = tvm.DataType('float4_e2m1fn')
    float4_e2m1fnx2 = tvm.DataType('float4_e2m1fnx2')
    float4_e2m1fnx4 = tvm.DataType('float4_e2m1fnx4')
    float4_e2m1fnx8 = tvm.DataType('float4_e2m1fnx8')
    float4_e2m1fnx16 = tvm.DataType('float4_e2m1fnx16')
    float4_e2m1fnx32 = tvm.DataType('float4_e2m1fnx32')
    float4_e2m1fnx64 = tvm.DataType('float4_e2m1fnx64')
    bfloat16 = tvm.DataType('bfloat16')

_all_dtypes = [
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
]

__all__ = _all_dtypes + [
    'AnyDType', 'get_tvm_dtype',
]
