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
    Annotated,
    Any,
    Generic,
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

if TYPE_CHECKING:

    class dyn[_T](tir.Var):
        dtype: tvm.DataType

    _Shape = TypeVarTuple("_Shape", bound=int)
    _Stride = TypeVar("_Stride", Tuple[int | dyn[int], ...])
    class BaseTensor(Generic[*_Shape, _Stride], tir.Buffer):
        shape: Tuple[tir.PrimExpr, ...]
        strides: Tuple[tir.PrimExpr, ...]
        dtype: tvm.DataType
        arg_idx: Optional[int] = None

        def get_shape(self) -> Tuple[*_Shape]: ...
        def get_strides(self) -> _Stride: ...
        def params(self) -> Tuple[*_Shape, tvm.DataType] : ...
        def all_params(self) -> Tuple[Tuple[*_Shape], _Stride, tvm.DataType]: ...

    _ShapeTup = TypeVar('_Shape', Tuple[int | dyn[int], ...])
    class StridedTensor(Generic[_ShapeTup, _Stride], BaseTensor[*_ShapeTup, _Stride]):
        pass

    class Tensor(BaseTensor[*_Shapes, Tuple[int | dyn[int], ...]]):
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
        return self.data.__eq__(rhs.data)
    def __ne__(self, rhs) -> bool:
        return self.data.__ne__(rhs.data)


@dataclass(frozen=True, slots=True)
class TuneMany[_T]:
    data: Tuple[_T, ...]
    def __hash__(self) -> int:
        return id(self.data)
    def __eq__(self, rhs) -> bool:
        return self.data is rhs.data
    def __ne__(self, rhs) -> bool:
        return self.data is not rhs.data


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


def place(*shape: Tuple[int, ...], dtype: AnyDType, strides: Optional[Tuple[int, ...]] = None, device: Optional[torch.device] = None) -> Place:
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


@dataclass(frozen=True, slots=True)
class _param:
    data: Any

def _apply_tvm_patches():
    def _as_value(v):
        if isinstance(v, tir.expr.ConstExpr):
            return v.value
        return v
    def _as_value_tup(v):
        return tuple(map(lambda x: _as_value(x), v))
    def _as_param_tup(v):
        return tuple(map(lambda x: _param(_as_value(x)), v))
    def __buf_params(self):
        return _as_param_tup((*self.shape, self.dtype))
    def __buf_all_params(self):
        return _as_param_tup(self.shape), _as_param_tup(self.strides), _param(_as_value(self.dtype))
    def __buf_get_shape(self):
        return _as_value_tup(self.shape)
    def __buf_get_strides(self):
        return _as_value_tup(self.strides)

    tir.Buffer.params = __buf_params
    tir.Buffer.all_params = __buf_all_params
    tir.Buffer.get_shape = __buf_get_shape
    tir.Buffer.get_strides = __buf_get_strides

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
