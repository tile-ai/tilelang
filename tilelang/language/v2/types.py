from __future__ import annotations
from dataclasses import dataclass
from tilelang import tvm
from tvm import tir, ir
import torch
import ctypes
from typing import (
    Any,
    Tuple,
    TypeVar,
    Iterable,
    Optional,
    TypeVarTuple,
    TYPE_CHECKING,
)


@dataclass(frozen=True, slots=True)
class DynSchema:
    ty: Optional[type] = int
    name: Optional[str] = None

    def __getitem__(self, params):
        if isinstance(params, tuple):
            ty, name = params
            return DynSchema(ty=ty, name=name)
        return DynSchema(ty=params, name=self.name)


@dataclass(frozen=True, slots=True)
class ConstSchema:
    def __getitem__(self, params):
        return self


@dataclass(frozen=True, slots=True)
class StridedTensorSchema:
    shape: Optional[Tuple[type, ...]] = None
    stride: Optional[Tuple[type, ...]] = None

    def __getitem__(self, params):
        if isinstance(params, tuple):
            shape, stride = params
            return StridedTensorSchema(shape=shape, stride=stride)
        return StridedTensorSchema(shape=params, stride=self.stride)


class TensorSchema(StridedTensorSchema):
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        stride = [int for _ in range(len(params))]
        dyn_flag = False
        for i in reversed(range(len(params))):
            if dyn_flag:
                stride[i] = dyn[int]
            if isinstance(params[i], DynSchema):
                dyn_flag = True
        return TensorSchema(shape=params, stride=stride)


_T = TypeVar("_T")
_Shapes = TypeVarTuple("_Shapes")

Schema = StridedTensorSchema | TensorSchema | DynSchema | ConstSchema

if TYPE_CHECKING:

    class dyn[_T](tir.Var):
        dtype: tvm.DataType

    class StridedTensor[_Shape, _Stride]:
        ptr: Any
        shape: _Shape
        stride: _Stride
        dtype: tvm.DataType

    class Tensor[*_Shapes](StridedTensor[Tuple[*_Shapes], Tuple[int | dyn[int], ...]]):
        pass

else:
    dyn = DynSchema()
    const = ConstSchema()
    StridedTensor = StridedTensorSchema()
    Tensor = TensorSchema()


@dataclass(frozen=True, slots=True)
class BufferLike:
    buffer: tir.Buffer
    shape: Tuple[int | tir.PrimExpr, ...]
    stride: Tuple[int | tir.PrimExpr, ...]
    dtype: tvm.DataType
    arg_idx: Optional[int] = None
    device: Optional[torch.device] = None

    def offset_of(self, indices) -> tir.IntImm:
        return self.buffer.offset_of(indices)

    def __getitem__(self, indices):
        return self.buffer.__getitem__(indices)


@dataclass(frozen=True, slots=True)
class Tune[_T]:
    params: Tuple[_T]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, rhs) -> bool:
        return self is rhs

    def __ne__(self, rhs) -> bool:
        return self is not rhs


def tune(params: Iterable[_T]) -> Tune[_T]:
    assert len(params) > 0, "Expected a non-empty parameter candidates"
    return Tune(tuple(params))


@dataclass(frozen=True, slots=True)
class MakeEmpty:
    shape: Tuple[int | tir.Var, ...]
    stride: Tuple[int | tir.Var, ...]
    dtype: torch.dtype | tvm.DataType
    device: torch.device


def empty(
    shape: Tuple[*_Shapes],
    dtype: torch.dtype | tvm.DataType,
    device: torch.device = None,
) -> Tensor[*_Shapes]:
    prod = 1
    stride = [0 for _ in range(len(shape))]
    for i in reversed(range(len(shape))):
        stride[i] = prod
        prod = prod * shape[i]
    return MakeEmpty(shape=shape, stride=stride, dtype=dtype, device=device)


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


def cvt_dtype(ty: ir.Type | str | type | torch.dtype) -> tvm.DataType:
    if isinstance(ty, ir.Type):
        return ty
    if isinstance(ty, str):
        return tvm.DataType(ty)
    return tvm.DataType(_dtype_torch2tvm[ty])


def cvt_tvm_dtype_to_torch(ty: tvm.DataType | str) -> torch.dtype:
    if isinstance(ty, str):
        ty = tvm.DataType(ty)
    return _dtype_tvm2torch[ty]


def cvt_tvm_dtype_to_ctypes(ty: tvm.DataType | str) -> Any:
    if isinstance(ty, str):
        ty = tvm.DataType(ty)
    return _dtype_tvm2ctype[ty]


def cvt_tvm_dtype_to_cffi(ty: tvm.DataType | str) -> str:
    if isinstance(ty, str):
        ty = tvm.DataType(ty)
    return _dtype_tvm2cffi[ty]


def get_ptr_type(
    ty: ir.Type | str | type | torch.dtype, scope: str = "global"
) -> ir.PointerType:
    ty = cvt_dtype(ty)
    return ir.PointerType(ir.PrimType(ty), scope)
