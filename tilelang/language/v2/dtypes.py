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
    (None, 'handle', ctypes.c_long, 'long'),  # use long to repr void*
    (bool, 'bool', ctypes.c_bool, 'bool'),
    (int, 'int32', ctypes.c_int32, 'int'),
    (float, 'float32', ctypes.c_float, 'float'),
    (torch.short, 'int16', ctypes.c_int16, 'short'),
    (torch.int, 'int32', ctypes.c_int32, 'int'),
    (torch.long, 'int64', ctypes.c_int64, 'long long'),
    (torch.half, 'float16', None, None),
    (torch.float, 'float32', ctypes.c_float, 'float'),
    (torch.double, 'float64', ctypes.c_double, 'double'),

    #   (pytype,                'tvm dtype str',    'ctypes dtype',     'cffi dtype')
    (torch.bool, 'bool', ctypes.c_bool, 'bool'),
    (torch.int8, 'int8', ctypes.c_int8, 'char'),
    (torch.int16, 'int16', ctypes.c_int16, 'short'),
    (torch.int32, 'int32', ctypes.c_int32, 'int'),
    (torch.int64, 'int64', ctypes.c_int64, 'long long'),
    (torch.uint8, 'uint8', ctypes.c_uint8, 'unsigned char'),
    (torch.uint16, 'uint16', ctypes.c_uint16, 'unsigned short'),
    (torch.uint32, 'uint32', ctypes.c_uint32, 'unsigned int'),
    (torch.uint64, 'uint64', ctypes.c_uint64, 'unsigned long long'),
    (torch.float16, 'float16', None, None),
    (torch.float32, 'float32', ctypes.c_float, 'float'),
    (torch.float64, 'float64', ctypes.c_double, 'double'),
    (torch.float8_e4m3fn, 'float8_e4m3fn', None, None),
    (torch.float8_e4m3fnuz, 'float8_e4m3fnuz', None, None),
    (torch.float8_e5m2, 'float8_e5m2', None, None),
    (torch.float8_e5m2fnuz, 'float8_e5m2fnuz', None, None),
    (torch.float8_e8m0fnu, 'float8_e8m0fnu', None, None),
    (torch.bfloat16, 'bfloat16', None, None),
]


def _create_type_mapper(sidx, didx, smapper=lambda x: x, dmapper=lambda x: x):
    return {
        smapper(item[sidx]): dmapper(item[didx])
        for item in _dtype_cvt
        if item[didx] is not None and item[sidx] is not None
    }


_dtype_tvm2py = _create_type_mapper(1, 0, lambda x: tvm.DataType(x))
_dtype_tvm2ctype = _create_type_mapper(1, 2, lambda x: tvm.DataType(x))
_dtype_tvm2cffi = _create_type_mapper(1, 3, lambda x: tvm.DataType(x))


class dtype:
    __cvt = _create_type_mapper(0, 1)

    def __init__(self, value: AnyDType):
        if isinstance(value, dtype):
            value = value.name
        if not isinstance(value, str):
            if value not in self.__cvt:
                raise TypeError(
                    f"Unsupported dtype: {value}, expected one of {list(self.__cvt.keys())}")
            value = self.__cvt[value]
        self.name = value

    def __eq__(self, other: AnyDType):
        if isinstance(other, str):
            return str.__eq__(self.name, other)
        if other in self.__cvt:
            return str.__eq__(self.name, self.__cvt[other])
        return NotImplemented

    def __req__(self, other: AnyDType):
        if isinstance(other, str):
            return str.__eq__(self.name, other)
        if other in self.__cvt:
            return str.__eq__(self.name, self.__cvt[other])
        return NotImplemented

    def __ne__(self, other: AnyDType):
        if isinstance(other, str):
            return str.__ne__(self.name, other)
        if other in self.__cvt:
            return str.__ne__(self.name, self.__cvt[other])
        return NotImplemented

    def __rne__(self, other: AnyDType):
        if isinstance(other, str):
            return str.__ne__(self.name, other)
        if other in self.__cvt:
            return str.__ne__(self.name, self.__cvt[other])
        return NotImplemented

    def __repr__(self):
        return f"dtype({str.__repr__(self.name)})"

    def __hash__(self):
        return str.__hash__(self.name)

    def __call__(self, expr=None, is_size_var: bool = False) -> tir.Var:
        return getattr(tb_ffi, self.name.title())(expr, is_size_var)

    def get_tvm_dtype(self) -> tvm.DataType:
        return tvm.DataType(self.name)


def get_tvm_dtype(value: AnyDType) -> tvm.DataType:
    if isinstance(value, (tvm.DataType, ir.Type)):
        return value
    if isinstance(value, dtype):
        return value.get_tvm_dtype()
    return dtype(value).get_tvm_dtype()


if TYPE_CHECKING:

    class int8(dtype):
        ...

    class int16(dtype):
        ...

    class int32(dtype):
        ...

    class int64(dtype):
        ...

    class uint8(dtype):
        ...

    class uint16(dtype):
        ...

    class uint32(dtype):
        ...

    class uint64(dtype):
        ...

    class float16(dtype):
        ...

    class float32(dtype):
        ...

    class float64(dtype):
        ...

    class bool(dtype):
        ...

    class float8_e4m3fn(dtype):
        ...

    class float8_e4m3fnuz(dtype):
        ...

    class float8_e5m2(dtype):
        ...

    class float8_e5m2fnuz(dtype):
        ...

    class float8_e8m0fnu(dtype):
        ...

    class bfloat16(dtype):
        ...
else:
    int8 = dtype('int8')
    int16 = dtype('int16')
    int32 = dtype('int32')
    int64 = dtype('int64')
    uint8 = dtype('uint8')
    uint16 = dtype('uint16')
    uint32 = dtype('uint32')
    uint64 = dtype('uint64')
    float16 = dtype('float16')
    float32 = dtype('float32')
    float64 = dtype('float64')
    bool = dtype('bool')
    float8_e4m3fn = dtype('float8_e4m3fn')
    float8_e4m3fnuz = dtype('float8_e4m3fnuz')
    float8_e5m2 = dtype('float8_e5m2')
    float8_e5m2fnuz = dtype('float8_e5m2fnuz')
    float8_e8m0fnu = dtype('float8_e8m0fnu')
    bfloat16 = dtype('bfloat16')
