from __future__ import annotations
from dataclasses import dataclass
from tilelang import tvm
from tilelang.language.dtypes import VoidPtr
from tvm import tir
import torch
from tilelang.language.dtypes import AnyDType
from typing import (
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


@dataclass(frozen=True, slots=True)
class BufferLike:
    buffer: tir.Buffer
    shape: Tuple[int | tir.PrimExpr, ...]
    stride: Tuple[int | tir.PrimExpr, ...]
    dtype: tvm.DataType
    arg_idx: Optional[int] = None
    # device: Optional[Device] = None

    def offset_of(self, indices) -> tir.IntImm:
        return self.buffer.offset_of(indices)

    def __getitem__(self, indices):
        return self.buffer.__getitem__(indices)

# @dataclass(frozen=True, slots=True)
# class DeviceSchema:
#     pass

_T = TypeVar("_T")
_Shapes = TypeVarTuple("_Shapes")

Schema = StridedTensorSchema | TensorSchema | DynSchema | ConstSchema

if TYPE_CHECKING:

    class dyn[_T](tir.Var):
        dtype: tvm.DataType

    class StridedTensor[_Shape, _Stride]:
        shape: _Shape
        stride: _Stride
        dtype: tvm.DataType
        # device: Device

    class Tensor[*_Shapes](StridedTensor[Tuple[*_Shapes], Tuple[int | dyn[int], ...]]):
        pass

    ptr = dyn[VoidPtr]

else:
    dyn = DynSchema()
    const = ConstSchema()
    StridedTensor = StridedTensorSchema()
    Tensor = TensorSchema()
    ptr = DynSchema(ty=VoidPtr)


@dataclass(frozen=True, slots=True)
class MakeEmpty:
    shape: Tuple[int | tir.Var, ...]
    stride: Tuple[int | tir.Var, ...]
    dtype: torch.dtype | tvm.DataType
    # device: torch.device


def empty(
    shape: Tuple[*_Shapes],
    dtype: torch.dtype | tvm.DataType,
    # device: Optional[Device] = None,
) -> Tensor[*_Shapes]:
    prod = 1
    stride = [0 for _ in range(len(shape))]
    for i in reversed(range(len(shape))):
        stride[i] = prod
        prod = prod * shape[i]
    return MakeEmpty(shape=shape, stride=stride, dtype=dtype)#, device=device)


@dataclass(frozen=True, slots=True)
class Tune[_T]:
    data: Tuple[_T, ...]
    def __hash__(self) -> int:
        return self.data.__hash__()
    def __eq__(self, rhs) -> bool:
        return self.data.__eq__(rhs)
    def __ne__(self, rhs) -> bool:
        return self.data.__ne__(rhs)


@dataclass(frozen=True, slots=True)
class TuneMany[_T]:
    data: Tuple[_T, ...]
    def __hash__(self) -> int:
        return id(self.data)
    def __eq__(self, rhs) -> bool:
        return self.data is rhs.data
    def __ne__(self, rhs) -> bool:
        return self.data is not rhs.data


def tune(params: Iterable[_T]) -> Tune[_T] | TuneMany[_T]:
    params = tuple(params)
    assert len(params) > 0, "Expected a non-empty parameter candidates"
    if len(params) > 8:
        return TuneMany(params)
    else:
        return Tune(params)


@dataclass
class Place:
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: AnyDType
    device: torch.device

    def data_ptr(self):
        raise RuntimeError("Trying to call kernel on a place holder.")

    def stride(self) -> Tuple[int, ...]:
        return self.strides


def place(shape: Tuple[int, ...], dtype: AnyDType, strides: Optional[Tuple[int, ...]] = None, device: Optional[torch.device] = None) -> Place:
    if strides is None:
        prod = 1
        strides = [0 for _ in range(len(shape))]
        for i in reversed(range(len(shape))):
            strides[i] = prod
            prod = prod * shape[i]
        strides = tuple(strides)
    if device is None:
        device = torch.cuda.current_device()
    return Place(shape=shape, strides=strides, dtype=dtype, device=device)

