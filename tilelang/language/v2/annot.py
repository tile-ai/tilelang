from __future__ import annotations
from dataclasses import dataclass
from types import GenericAlias

from tvm import tir
from tvm.script.ir_builder.tir import buffer, handle, match_buffer
from typing import Callable, TypeVar, ParamSpec, Generic, TypeVarTuple, Unpack, TYPE_CHECKING, get_origin, get_args
from .dtypes import AnyDType
from . import dtypes as dt


_Shapes = TypeVarTuple('_Shapes')
_Shape = ParamSpec('_Shape')
_Strides = TypeVarTuple('_Strides')
_Stride = ParamSpec('_Stride')
_DType = TypeVar('_DType')


class BufferProxy(Generic[_Shape, _DType], tir.Buffer):
    _default_scope = 'global'

    def __new__(cls, 
        shape: tuple[Unpack[_Shapes]],
        dtype: _DType="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope=None,
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> Tensor[Callable[[Unpack[_Shapes]]], _DType]:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope or cls._default_scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    if TYPE_CHECKING:
        @property
        def shape(self: Tensor[Callable[[Unpack[_Shapes]]], _DType]) -> tuple[Unpack[_Shapes]]: ...

        @property
        def strides(self) -> tuple[tir.PrimExpr | tir.Var, ...]: ...

        @property
        def dtype(self) -> dt.dtype[_DType]: ...

    

class StridedTensor(Generic[_Shape, _Stride, _DType], BufferProxy[_Shape, _DType]):
    _default_scope = 'global'

    def __new__(
            cls,
            shape: tuple[Unpack[_Shapes]],
            strides: tuple[Unpack[_Strides]],
            dtype: _DType = 'float32',
    ) -> StridedTensor[Callable[[Unpack[_Shapes]]], Callable[[Unpack[_Strides]]], _DType]:
        return super().__new__(cls, shape, dtype, strides=strides)


class Tensor(Generic[_Shape, _DType], BufferProxy[_Shape, _DType]):
    _default_scope = 'global'


class FragmentBuffer(Generic[_Shape, _DType], BufferProxy[_Shape, _DType]):
    _default_scope = 'local.fragment'


class SharedBuffer(Generic[_Shape, _DType], BufferProxy[_Shape, _DType]):
    _default_scope = 'shared.dyn'


class LocalBuffer(Generic[_Shape, _DType], BufferProxy[_Shape, _DType]):
    _default_scope = 'local'


class dyn(tir.Var):

    if TYPE_CHECKING:
        @property
        def dtype(self) -> dt.dtype[_DType]: ...


class AnnotParser:
    def __init__(self):
        self.args = []
        self.extra_vars = []
        self.named_vars = {}

    def get_or_create_var(self, name: str, dtype: AnyDType):
        if name not in self.named_vars:
            self.named_vars[name] = tir.Var(name, dtype)
        return self.named_vars[name]

    def convert_shape_annot(self, annots):
        pass

    def parse_annot(self, annot):
        if isinstance(annot, tir.Var):
            self.args.append(annot)
        if isinstance(annot, tir.Buffer):
            for s in annot.shape:
                if isinstance(s, tir.Var):
                    self.extra_vars.append(s)


def foo(
        A: Tensor[[int, dyn], dt.float32],
):
    N, M = A.shape
    ty = A.dtype
