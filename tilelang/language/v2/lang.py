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
    Any,
    Unpack,
    Generic,
    Optional,
    overload,
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
        raise SyntaxError("Expected a tuple of (shape, stride)")


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


@dataclass(frozen=True, slots=True)
class _param:
    data: Any


def _as_value(v):
    if isinstance(v, tir.expr.ConstExpr):
        return v.value
    return v


def _as_value_tup(v):
    return tuple(map(lambda x: _as_value(x), v))


def _as_param_tup(v):
    return tuple(map(lambda x: _param(_as_value(x)), v))


@dataclass(frozen=True, slots=True)
class TensorV2:
    buffer: tir.Buffer
    arg_idx: Optional[int] = None

    @property
    def name(self) -> str:
        return self.buffer.name

    @property
    def shape(self) -> "Tuple[int | tir.Var, ...]":
        return _as_value_tup(self.buffer.shape)

    @property
    def strides(self) -> "Tuple[int | tir.Var, ...]":
        return _as_value_tup(self.buffer.strides)

    @property
    def dtype(self) -> tvm.DataType:
        return self.buffer.dtype

    def shape_params(self) -> Tuple[int | tir.Var, ...]:
        return _as_param_tup(self.buffer.shape)

    def stride_params(self) -> Tuple[int | tir.Var, ...]:
        return _as_param_tup(self.buffer.strides)

    @overload
    def params(self) -> "Tuple[*Tuple[int | tir.Var, ...], tvm.DataType]":
        ...

    @overload
    def all_params(
        self
    ) -> "Tuple[Tuple[*Tuple[int | tir.Var, ...]], Tuple[int | tir.Var, ...], tvm.DataType]":
        ...

    def params(self):
        return *self.shape_params(), self.dtype

    def all_params(self):
        return self.shape_params(), self.stride_params(), self.dtype

    def __getitem__(self, idx):
        return self.buffer.__getitem__(idx)


if TYPE_CHECKING:

    class dyn(Generic[_T], tir.Var):
        dtype: tvm.DataType

    _Shape = TypeVarTuple("_Shape")
    _Stride = TypeVar("_Stride", Tuple[int, ...])

    class _BaseTensor(Generic[Unpack[_Shape], _Stride], tir.Buffer):
        name: str
        shape: "Tuple[*_Shape]"
        strides: _Stride
        dtype: tvm.DataType
        arg_idx: Optional[int] = None

        def shape_params(self) -> "Tuple[*_Shape]":
            ...

        def stride_params(self) -> _Stride:
            ...

        def params(self) -> "Tuple[*_Shape, tvm.DataType]":
            ...

        def all_params(self) -> "Tuple[Tuple[*_Shape], _Stride, tvm.DataType]":
            ...

    _ShapeTup = TypeVar('_Shape', Tuple[Any, ...])

    class StridedTensor(Generic[_ShapeTup, _Stride], _BaseTensor[Unpack[_ShapeTup], _Stride]):
        pass

    class Tensor(_BaseTensor[Unpack[_Shapes], Tuple[Any, ...]]):
        pass

    ptr = dyn[VoidPtr]

else:
    dyn = DynSchema()
    const = ConstSchema()
    StridedTensor = StridedTensorSchema()
    Tensor = TensorSchema()
    ptr = DynSchema(ty=VoidPtr)

Tensor1D = Tensor[int]
Tensor2D = Tensor[int, int]
Tensor3D = Tensor[int, int, int]
Tensor4D = Tensor[int, int, int, int]


@dataclass(frozen=True, slots=True)
class MakeEmpty:
    shape: Tuple[int | tir.Var, ...]
    stride: Tuple[int | tir.Var, ...]
    dtype: torch.dtype | tvm.DataType
    # device: torch.device


def empty(
    shape: "Tuple[*_Shapes]",
    dtype: torch.dtype | tvm.DataType,
    # device: Optional[Device] = None,
) -> "Tensor[*_Shapes]":
    prod = 1
    stride = [0 for _ in range(len(shape))]
    for i in reversed(range(len(shape))):
        stride[i] = prod
        prod = prod * shape[i]
    return MakeEmpty(shape=shape, stride=stride, dtype=dtype)  #, device=device)


@dataclass(frozen=True, slots=True)
class Tune(Generic[_T]):
    data: Tuple[_T, ...]

    def __hash__(self) -> int:
        return self.data.__hash__()

    def __eq__(self, rhs) -> bool:
        return isinstance(rhs, Tune) and self.data == rhs.data

    def __ne__(self, rhs) -> bool:
        return isinstance(rhs, Tune) and self.data == rhs.data


@dataclass(frozen=True, slots=True)
class TuneMany(Generic[_T]):
    data: Tuple[_T, ...]

    def __hash__(self) -> int:
        return id(self.data)

    def __eq__(self, rhs) -> bool:
        return isinstance(rhs, TuneMany) and self.data is rhs.data

    def __ne__(self, rhs) -> bool:
        return isinstance(rhs, TuneMany) and self.data is not rhs.data


def tune(params: Iterable[_T], tune_many_threshold: int = 10) -> Tune[_T] | TuneMany[_T]:
    params = tuple(params)
    assert len(params) > 0, "Expected a non-empty parameter candidates"
    if len(params) > tune_many_threshold:
        return TuneMany(params)
    else:
        return Tune(params)


class empty_data_ptr:

    def __repr__(self):
        return "empty_data_ptr()"


@dataclass
class Place:
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: AnyDType
    device: torch.device

    def data_ptr(self):
        return empty_data_ptr()

    def stride(self) -> Tuple[int, ...]:
        return self.strides


def place(*shape: Tuple[int, ...],
          dtype: AnyDType,
          strides: Optional[Tuple[int, ...]] = None,
          device: Optional[torch.device] = None) -> Place:
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


# monkey patches

_tvm_patched = False


def _apply_tvm_patches():

    def __array_eq(self, rhs):
        if isinstance(rhs, tuple):
            return tuple(self) == rhs
        if isinstance(rhs, list):
            return list(self) == rhs
        if isinstance(rhs, tvm.ffi.container.Array):
            return tvm.core.Object.__eq__(self, rhs)

    def __array_ne(self, rhs):
        if isinstance(rhs, tuple):
            return tuple(self) != rhs
        if isinstance(rhs, list):
            return list(self) != rhs
        if isinstance(rhs, tvm.ffi.container.Array):
            return tvm.core.Object.__ne__(self, rhs)

    def __array_req(self, lhs):
        if isinstance(lhs, tuple):
            return tuple(self) == lhs
        if isinstance(lhs, list):
            return list(self) == lhs
        if isinstance(lhs, tvm.ffi.container.Array):
            return tvm.core.Object.__eq__(self, lhs)

    def __array_rne(self, rhs):
        if isinstance(rhs, tuple):
            return tuple(self) != rhs
        if isinstance(rhs, list):
            return list(self) != rhs
        if isinstance(rhs, tvm.ffi.container.Array):
            return tvm.core.Object.__ne__(self, rhs)

    tvm.ffi.container.Array.__eq__ = __array_eq
    tvm.ffi.container.Array.__ne__ = __array_ne
    tvm.ffi.container.Array.__req__ = __array_req
    tvm.ffi.container.Array.__rne__ = __array_rne


if not _tvm_patched:
    _tvm_patched = True
    _apply_tvm_patches()
