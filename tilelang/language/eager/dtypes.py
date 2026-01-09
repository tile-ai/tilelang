from tilelang import tvm
from tvm import ir
import torch
from typing import Generic, TypeVar, Union, TYPE_CHECKING
from tvm import tir
import tvm.script.ir_builder.tir._ffi_api as tb_ffi
import numpy as np
from tilelang import logger

_T = TypeVar("_T")

if TYPE_CHECKING:

    class dtype(Generic[_T]):
        def as_torch(self) -> torch.dtype: ...
else:
    dtype = tvm.DataType

# Python 3.9 compatibility: avoid PEP 604 unions at runtime
AnyDType = Union[ir.Type, str, type, torch.dtype, dtype]

_PYTHON_DTYPE_TO_STR = {
    bool: "bool",
    int: "int32",
    float: "float32",
}

_NUMPY_DTYPE_TO_STR = {
    np.bool_: "bool",
    np.short: "int16",
    np.int_: "int64",
    np.longlong: "int64",
    np.half: "float16",
    np.double: "float64",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
}

_NUMPY_DTYPE_TO_STR.update({np.dtype(k): v for k, v in _NUMPY_DTYPE_TO_STR.items()})

_TORCH_DTYPE_TO_STR = {
    torch.bool: "bool",
    torch.short: "int16",
    torch.int: "int32",
    torch.long: "int64",
    torch.half: "float16",
    torch.float: "float32",
    torch.double: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}

_extended_torch_dtypes = [
    ("float8_e4m3fn",),
    ("float8_e4m3fnuz",),
    ("float8_e5m2",),
    ("float8_e5m2fnuz",),
    ("float8_e8m0fnu",),
    ("float4_e2m1fnx2",),
]
for dtype_name_tuple in _extended_torch_dtypes:
    dtype_name = dtype_name_tuple[0]
    torch_dtype = None
    if dtype_name == "float4_e2m1fnx2":
        torch_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    else:
        torch_dtype = getattr(torch, dtype_name, None)

    if torch_dtype is not None:
        _TORCH_DTYPE_TO_STR[torch_dtype] = dtype_name


_CANONICAL_TO_DISPLAY_STR = {
    "double": "float64",
    "float": "float32",
    "int": "int32",
    "long": "int64",
    "short": "int16",
    "uint": "uint32",
    "ulong": "uint64",
}

_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_STR.items()}

# _STR_TO_NUMPY_DTYPE = {v: k for k, v in _NUMPY_DTYPE_TO_STR.items()}

_DTYPE_TO_STR = {**_PYTHON_DTYPE_TO_STR, **_NUMPY_DTYPE_TO_STR, **_TORCH_DTYPE_TO_STR}

_STR_TO_TVM_DTYPE_CALL = {
    "bool": "Boolean",
    "int4": "Int4",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "float16": "Float16",
    "float32": "Float32",
    "float64": "Float64",
    "bfloat16": "BFloat16",
    "float8_e4m3": "Float8E4M3",
    "float8_e4m3fn": "Float8E4M3FN",
    "float8_e4m3fnuz": "Float8E4M3FNUZ",
    "float8_e5m2": "Float8E5M2",
    "float8_e5m2fnuz": "Float8E5M2FNUZ",
    "float8_e8m0fnu": "Float8E8M0FNU",
}

int_ = int


def __dtype_call__(self: dtype, expr=None, is_size_var: bool = False) -> tir.Var:
    if isinstance(expr, int_):
        return tvm.tir.const(expr, dtype=self)
    if self in _STR_TO_TVM_DTYPE_CALL:
        attr = _STR_TO_TVM_DTYPE_CALL[self]
        call = getattr(tb_ffi, attr, None)
        return call(expr, is_size_var)
    # try to construct the ffi call
    if self.startswith("uint"):
        val = "UInt" + self[4:]
    elif self.startswith("int"):
        val = "Int" + self[3:]
    elif self.startswith("float"):
        val = "Float" + self[5:]
    elif self.startswith("bfloat"):
        val = "BFloat" + self[6:]
    else:
        raise TypeError(f"Invalid type {self}")
    if "_" in val:
        first, second = val.split("_", maxsplit=1)
        val = first + second.upper()
    call = getattr(tb_ffi, val, None)
    if call is None:
        raise TypeError(
            f"Convert to datatype `{self}` is not supported by tvm\ncalling failed on `tvm.script.ir_builder.tir._ffi_api.{val}`"
        )
    return call(expr, is_size_var)


def __dtype_as_torch__(self: dtype) -> torch.dtype:
    """Convert TileLang dtype to PyTorch dtype."""
    dtype_str = str(self)

    if dtype_str == "float8_e4m3":
        # Check if we're on HIP (AMD ROCm) or CUDA
        if torch.version.hip is not None:
            # HIP backend - use float8_e4m3fnuz
            assert hasattr(torch, "float8_e4m3fnuz"), (
                "torch.float8_e4m3fnuz is not supported in this version of torch. Please upgrade torch >= 2.2.0"
            )
            return torch.float8_e4m3fnuz
        else:
            # CUDA backend - use float8_e4m3fn
            assert hasattr(torch, "float8_e4m3fn"), (
                "torch.float8_e4m3fn is not supported in this version of torch. Please upgrade torch >= 2.1.0"
            )
            return torch.float8_e4m3fn
    elif dtype_str == "float8_e5m2":
        assert hasattr(torch, "float8_e5m2"), "torch.float8_e5m2 is not supported in this version of torch. Please upgrade torch >= 2.1.0"
        return torch.float8_e5m2
    elif dtype_str == "float8_e4m3fnuz":
        assert hasattr(torch, "float8_e4m3fnuz"), (
            "torch.float8_e4m3fnuz is not supported in this version of torch. Please upgrade torch >= 2.2.0"
        )
        return torch.float8_e4m3fnuz
    elif dtype_str == "float8_e8m0fnu":
        assert hasattr(torch, "float8_e8m0fnu"), (
            "torch.float8_e8m0fnu is not supported in this version of torch. Please upgrade torch >= 2.8.0"
        )
        return torch.float8_e8m0fnu
    elif dtype_str == "float4_e2m1fnx2":
        assert hasattr(torch, "float4_e2m1fnx2"), (
            "torch.float4_e2m1fnx2 is not supported in this version of torch. Please upgrade torch >= 2.8.0"
        )
        return torch.float4_e2m1fn_x2
    elif dtype_str == "float4_e2m1fn":
        logger.info("torch doesn't support float4_e2m1fn, using float4_e2m1fnx2 as storage dtype.")
        return torch.float4_e2m1fn_x2 if hasattr(torch, "float4_e2m1fn_x2") else torch.int8
    elif dtype_str in _STR_TO_TORCH_DTYPE:
        return _STR_TO_TORCH_DTYPE[dtype_str]

    raise ValueError(f"Cannot convert dtype '{dtype_str}' to torch.dtype. Supported dtypes: {list(_STR_TO_TORCH_DTYPE.keys())}")


__orig_dtype_new = dtype.__new__


def __dtype_new__(cls, value: AnyDType) -> dtype:
    if isinstance(value, str):
        return __orig_dtype_new(cls, _CANONICAL_TO_DISPLAY_STR.get(value, value))
    elif value in _DTYPE_TO_STR:
        return __orig_dtype_new(cls, _DTYPE_TO_STR[value])
    else:
        expected = set(list(_DTYPE_TO_STR.keys()) + list(_DTYPE_TO_STR.values()))
        raise TypeError(f"Invalid DataType {value}({type(value)}), expect one of {expected}")


dtype.__call__ = __dtype_call__
dtype.__new__ = __dtype_new__
dtype.as_torch = __dtype_as_torch__


def get_tvm_dtype(value: AnyDType) -> dtype:
    if isinstance(value, (dtype, ir.Type)):
        return value
    return dtype(value)


if TYPE_CHECKING:
    # yapf: disable
    class bool(dtype): ...
    class short(dtype): ...
    class int(dtype): ...
    class uint(dtype): ...
    class long(dtype): ...
    class half(dtype): ...
    class float(dtype): ...
    class double(dtype): ...
    class int4(dtype): ...
    class int8(dtype): ...
    class int16(dtype): ...
    class int32(dtype): ...
    class int64(dtype): ...
    class int8x2(dtype): ...
    class int16x2(dtype): ...
    class int32x2(dtype): ...
    class int64x2(dtype): ...
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
    class uint8x2(dtype): ...
    class uint16x2(dtype): ...
    class uint32x2(dtype): ...
    class uint64x2(dtype): ...
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
    bool = dtype("bool")
    short = dtype("int16")
    int = dtype("int32")
    uint = dtype("uint32")
    long = dtype("int64")
    half = dtype("float16")
    float = dtype("float32")
    double = dtype("float64")
    int4 = dtype("int4")
    int8 = dtype("int8")
    int16 = dtype("int16")
    int32 = dtype("int32")
    int64 = dtype("int64")
    int8x2 = dtype("int8x2")
    int16x2 = dtype("int16x2")
    int32x2 = dtype("int32x2")
    int64x2 = dtype("int64x2")
    int8x4 = dtype("int8x4")
    int16x4 = dtype("int16x4")
    int32x4 = dtype("int32x4")
    int64x4 = dtype("int64x4")
    int8x8 = dtype("int8x8")
    int16x8 = dtype("int16x8")
    int32x8 = dtype("int32x8")
    int64x8 = dtype("int64x8")
    int8x16 = dtype("int8x16")
    int16x16 = dtype("int16x16")
    int32x16 = dtype("int32x16")
    int64x16 = dtype("int64x16")
    int8x32 = dtype("int8x32")
    int16x32 = dtype("int16x32")
    int32x32 = dtype("int32x32")
    int64x32 = dtype("int64x32")
    int8x64 = dtype("int8x64")
    int16x64 = dtype("int16x64")
    int32x64 = dtype("int32x64")
    int64x64 = dtype("int64x64")
    uint8 = dtype("uint8")
    uint16 = dtype("uint16")
    uint32 = dtype("uint32")
    uint64 = dtype("uint64")
    uint8x2 = dtype("uint8x2")
    uint16x2 = dtype("uint16x2")
    uint32x2 = dtype("uint32x2")
    uint64x2 = dtype("uint64x2")
    uint8x4 = dtype("uint8x4")
    uint16x4 = dtype("uint16x4")
    uint32x4 = dtype("uint32x4")
    uint64x4 = dtype("uint64x4")
    uint8x8 = dtype("uint8x8")
    uint16x8 = dtype("uint16x8")
    uint32x8 = dtype("uint32x8")
    uint64x8 = dtype("uint64x8")
    uint8x16 = dtype("uint8x16")
    uint16x16 = dtype("uint16x16")
    uint32x16 = dtype("uint32x16")
    uint64x16 = dtype("uint64x16")
    uint8x32 = dtype("uint8x32")
    uint16x32 = dtype("uint16x32")
    uint32x32 = dtype("uint32x32")
    uint64x32 = dtype("uint64x32")
    uint8x64 = dtype("uint8x64")
    uint16x64 = dtype("uint16x64")
    uint32x64 = dtype("uint32x64")
    uint64x64 = dtype("uint64x64")
    float16 = dtype("float16")
    float32 = dtype("float32")
    float64 = dtype("float64")
    float16x2 = dtype("float16x2")
    float32x2 = dtype("float32x2")
    float64x2 = dtype("float64x2")
    float16x4 = dtype("float16x4")
    float32x4 = dtype("float32x4")
    float64x4 = dtype("float64x4")
    float16x8 = dtype("float16x8")
    float32x8 = dtype("float32x8")
    float64x8 = dtype("float64x8")
    float16x16 = dtype("float16x16")
    float32x16 = dtype("float32x16")
    float64x16 = dtype("float64x16")
    float16x32 = dtype("float16x32")
    float32x32 = dtype("float32x32")
    float64x32 = dtype("float64x32")
    float16x64 = dtype("float16x64")
    float32x64 = dtype("float32x64")
    float64x64 = dtype("float64x64")
    float8_e3m4 = dtype("float8_e3m4")
    float8_e3m4x2 = dtype("float8_e3m4x2")
    float8_e3m4x4 = dtype("float8_e3m4x4")
    float8_e3m4x8 = dtype("float8_e3m4x8")
    float8_e3m4x16 = dtype("float8_e3m4x16")
    float8_e3m4x32 = dtype("float8_e3m4x32")
    float8_e3m4x64 = dtype("float8_e3m4x64")
    float8_e4m3 = dtype("float8_e4m3")
    float8_e4m3x2 = dtype("float8_e4m3x2")
    float8_e4m3x4 = dtype("float8_e4m3x4")
    float8_e4m3x8 = dtype("float8_e4m3x8")
    float8_e4m3x16 = dtype("float8_e4m3x16")
    float8_e4m3x32 = dtype("float8_e4m3x32")
    float8_e4m3x64 = dtype("float8_e4m3x64")
    float8_e4m3b11fnuz = dtype("float8_e4m3b11fnuz")
    float8_e4m3b11fnuzx2 = dtype("float8_e4m3b11fnuzx2")
    float8_e4m3b11fnuzx4 = dtype("float8_e4m3b11fnuzx4")
    float8_e4m3b11fnuzx8 = dtype("float8_e4m3b11fnuzx8")
    float8_e4m3b11fnuzx16 = dtype("float8_e4m3b11fnuzx16")
    float8_e4m3b11fnuzx32 = dtype("float8_e4m3b11fnuzx32")
    float8_e4m3b11fnuzx64 = dtype("float8_e4m3b11fnuzx64")
    float8_e4m3fn = dtype("float8_e4m3fn")
    float8_e4m3fnx2 = dtype("float8_e4m3fnx2")
    float8_e4m3fnx4 = dtype("float8_e4m3fnx4")
    float8_e4m3fnx8 = dtype("float8_e4m3fnx8")
    float8_e4m3fnx16 = dtype("float8_e4m3fnx16")
    float8_e4m3fnx32 = dtype("float8_e4m3fnx32")
    float8_e4m3fnx64 = dtype("float8_e4m3fnx64")
    float8_e4m3fnuz = dtype("float8_e4m3fnuz")
    float8_e4m3fnuzx2 = dtype("float8_e4m3fnuzx2")
    float8_e4m3fnuzx4 = dtype("float8_e4m3fnuzx4")
    float8_e4m3fnuzx8 = dtype("float8_e4m3fnuzx8")
    float8_e4m3fnuzx16 = dtype("float8_e4m3fnuzx16")
    float8_e4m3fnuzx32 = dtype("float8_e4m3fnuzx32")
    float8_e4m3fnuzx64 = dtype("float8_e4m3fnuzx64")
    float8_e5m2 = dtype("float8_e5m2")
    float8_e5m2x2 = dtype("float8_e5m2x2")
    float8_e5m2x4 = dtype("float8_e5m2x4")
    float8_e5m2x8 = dtype("float8_e5m2x8")
    float8_e5m2x16 = dtype("float8_e5m2x16")
    float8_e5m2x32 = dtype("float8_e5m2x32")
    float8_e5m2x64 = dtype("float8_e5m2x64")
    float8_e5m2fnuz = dtype("float8_e5m2fnuz")
    float8_e5m2fnuzx2 = dtype("float8_e5m2fnuzx2")
    float8_e5m2fnuzx4 = dtype("float8_e5m2fnuzx4")
    float8_e5m2fnuzx8 = dtype("float8_e5m2fnuzx8")
    float8_e5m2fnuzx16 = dtype("float8_e5m2fnuzx16")
    float8_e5m2fnuzx32 = dtype("float8_e5m2fnuzx32")
    float8_e5m2fnuzx64 = dtype("float8_e5m2fnuzx64")
    float8_e8m0fnu = dtype("float8_e8m0fnu")
    float8_e8m0fnux2 = dtype("float8_e8m0fnux2")
    float8_e8m0fnux4 = dtype("float8_e8m0fnux4")
    float8_e8m0fnux8 = dtype("float8_e8m0fnux8")
    float8_e8m0fnux16 = dtype("float8_e8m0fnux16")
    float8_e8m0fnux32 = dtype("float8_e8m0fnux32")
    float8_e8m0fnux64 = dtype("float8_e8m0fnux64")
    float6_e2m3fn = dtype("float6_e2m3fn")
    float6_e2m3fnx2 = dtype("float6_e2m3fnx2")
    float6_e2m3fnx4 = dtype("float6_e2m3fnx4")
    float6_e2m3fnx8 = dtype("float6_e2m3fnx8")
    float6_e2m3fnx16 = dtype("float6_e2m3fnx16")
    float6_e2m3fnx32 = dtype("float6_e2m3fnx32")
    float6_e2m3fnx64 = dtype("float6_e2m3fnx64")
    float6_e3m2fn = dtype("float6_e3m2fn")
    float6_e3m2fnx2 = dtype("float6_e3m2fnx2")
    float6_e3m2fnx4 = dtype("float6_e3m2fnx4")
    float6_e3m2fnx8 = dtype("float6_e3m2fnx8")
    float6_e3m2fnx16 = dtype("float6_e3m2fnx16")
    float6_e3m2fnx32 = dtype("float6_e3m2fnx32")
    float6_e3m2fnx64 = dtype("float6_e3m2fnx64")
    float4_e2m1fn = dtype("float4_e2m1fn")
    float4_e2m1fnx2 = dtype("float4_e2m1fnx2")
    float4_e2m1fnx4 = dtype("float4_e2m1fnx4")
    float4_e2m1fnx8 = dtype("float4_e2m1fnx8")
    float4_e2m1fnx16 = dtype("float4_e2m1fnx16")
    float4_e2m1fnx32 = dtype("float4_e2m1fnx32")
    float4_e2m1fnx64 = dtype("float4_e2m1fnx64")
    bfloat16 = dtype("bfloat16")

_all_dtypes = {
    "bool",
    "short",
    "int",
    "uint",
    "long",
    "half",
    "float",
    "double",
    "int4",
    "int8",
    "int16",
    "int32",
    "int64",
    "int8x2",
    "int16x2",
    "int32x2",
    "int64x2",
    "int8x4",
    "int16x4",
    "int32x4",
    "int64x4",
    "int8x8",
    "int16x8",
    "int32x8",
    "int64x8",
    "int8x16",
    "int16x16",
    "int32x16",
    "int64x16",
    "int8x32",
    "int16x32",
    "int32x32",
    "int64x32",
    "int8x64",
    "int16x64",
    "int32x64",
    "int64x64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint8x2",
    "uint16x2",
    "uint32x2",
    "uint64x2",
    "uint8x4",
    "uint16x4",
    "uint32x4",
    "uint64x4",
    "uint8x8",
    "uint16x8",
    "uint32x8",
    "uint64x8",
    "uint8x16",
    "uint16x16",
    "uint32x16",
    "uint64x16",
    "uint8x32",
    "uint16x32",
    "uint32x32",
    "uint64x32",
    "uint8x64",
    "uint16x64",
    "uint32x64",
    "uint64x64",
    "float16",
    "float32",
    "float64",
    "float16x2",
    "float32x2",
    "float64x2",
    "float16x4",
    "float32x4",
    "float64x4",
    "float16x8",
    "float32x8",
    "float64x8",
    "float16x16",
    "float32x16",
    "float64x16",
    "float16x32",
    "float32x32",
    "float64x32",
    "float16x64",
    "float32x64",
    "float64x64",
    "float8_e3m4",
    "float8_e3m4x2",
    "float8_e3m4x4",
    "float8_e3m4x8",
    "float8_e3m4x16",
    "float8_e3m4x32",
    "float8_e3m4x64",
    "float8_e4m3",
    "float8_e4m3x2",
    "float8_e4m3x4",
    "float8_e4m3x8",
    "float8_e4m3x16",
    "float8_e4m3x32",
    "float8_e4m3x64",
    "float8_e4m3b11fnuz",
    "float8_e4m3b11fnuzx2",
    "float8_e4m3b11fnuzx4",
    "float8_e4m3b11fnuzx8",
    "float8_e4m3b11fnuzx16",
    "float8_e4m3b11fnuzx32",
    "float8_e4m3b11fnuzx64",
    "float8_e4m3fn",
    "float8_e4m3fnx2",
    "float8_e4m3fnx4",
    "float8_e4m3fnx8",
    "float8_e4m3fnx16",
    "float8_e4m3fnx32",
    "float8_e4m3fnx64",
    "float8_e4m3fnuz",
    "float8_e4m3fnuzx2",
    "float8_e4m3fnuzx4",
    "float8_e4m3fnuzx8",
    "float8_e4m3fnuzx16",
    "float8_e4m3fnuzx32",
    "float8_e4m3fnuzx64",
    "float8_e5m2",
    "float8_e5m2x2",
    "float8_e5m2x4",
    "float8_e5m2x8",
    "float8_e5m2x16",
    "float8_e5m2x32",
    "float8_e5m2x64",
    "float8_e5m2fnuz",
    "float8_e5m2fnuzx2",
    "float8_e5m2fnuzx4",
    "float8_e5m2fnuzx8",
    "float8_e5m2fnuzx16",
    "float8_e5m2fnuzx32",
    "float8_e5m2fnuzx64",
    "float8_e8m0fnu",
    "float8_e8m0fnux2",
    "float8_e8m0fnux4",
    "float8_e8m0fnux8",
    "float8_e8m0fnux16",
    "float8_e8m0fnux32",
    "float8_e8m0fnux64",
    "float6_e2m3fn",
    "float6_e2m3fnx2",
    "float6_e2m3fnx4",
    "float6_e2m3fnx8",
    "float6_e2m3fnx16",
    "float6_e2m3fnx32",
    "float6_e2m3fnx64",
    "float6_e3m2fn",
    "float6_e3m2fnx2",
    "float6_e3m2fnx4",
    "float6_e3m2fnx8",
    "float6_e3m2fnx16",
    "float6_e3m2fnx32",
    "float6_e3m2fnx64",
    "float4_e2m1fn",
    "float4_e2m1fnx2",
    "float4_e2m1fnx4",
    "float4_e2m1fnx8",
    "float4_e2m1fnx16",
    "float4_e2m1fnx32",
    "float4_e2m1fnx64",
    "bfloat16",
}

__all__ = list(_all_dtypes) + [
    "dtype",
    "AnyDType",
    "get_tvm_dtype",
]
