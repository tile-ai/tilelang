from __future__ import annotations
from dataclasses import dataclass
from tilelang import tvm
from tvm import tir
import torch
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
