"""CuTe layout IR objects and layout-algebra Python API, in TileLang."""

from __future__ import annotations

from itertools import chain
from typing import Union
from collections.abc import Sequence

import tvm
import tvm_ffi
from tvm.ir import PrimExpr, Range
from tvm.ir.base import Node
from tvm.runtime import Scriptable

from tilelang._typing import BufferLikeType
from . import _cute_ffi_api
from .swizzle_mode import SwizzleMode

PyIntTuple = Union[int, PrimExpr, "ScaledBasis", tuple]
IntTupleLike = Union[int, PrimExpr, "ScaledBasis", Sequence, "IntTuple"]
ModeLike = int | Sequence[int]


def to_python(t: IntTuple) -> PyIntTuple:
    """Unpack an :class:`IntTuple` into Python: a branch becomes a nested ``tuple``, and each leaf becomes a plain ``int`` (static), a PrimExpr (dynamic), or a :class:`ScaledBasis` (basis stride)."""
    if isinstance(t, IntTupleTuple):
        return tuple(to_python(c) for c in t.fields)
    if isinstance(t, IntTupleConst):
        return int(t.value)
    if isinstance(t, IntTuplePrimExpr):
        return t.value
    if isinstance(t, IntTupleScaledBasis):
        return ScaledBasis(to_python(t.value), tuple(t.basis))
    raise TypeError(f"not an IntTuple: {t!r}")


def from_python(value: IntTupleLike) -> IntTuple:
    """Convert a Python value to an FFI :class:`IntTuple` (the inverse of :func:`to_python`): a nested tuple/list becomes an :class:`IntTupleTuple` branch; an int becomes :class:`IntTupleConst`; a PrimExpr becomes :class:`IntTuplePrimExpr`; a :class:`ScaledBasis` becomes :class:`IntTupleScaledBasis`; an already-built :class:`IntTuple` passes through."""
    if isinstance(value, IntTuple):
        return value
    if isinstance(value, (tuple, list)):
        return _cute_ffi_api.make_int_tuple_tuple([from_python(v) for v in value])
    if isinstance(value, int):
        return _cute_ffi_api.make_int_const(int(value))
    if isinstance(value, PrimExpr):
        return _cute_ffi_api.make_int_expr(value)
    if isinstance(value, ScaledBasis):
        return _cute_ffi_api.make_scaled_basis(from_python(value.value), list(value.mode))
    raise TypeError(f"cannot convert to IntTuple: {value!r}")


@tvm_ffi.register_object("tl.cute.Swizzle")
class Swizzle(Node, Scriptable):
    b_bits: int
    m_base: int
    s_shift: int

    @property
    def is_swizzled(self) -> bool:
        return _cute_ffi_api.swizzle_is_swizzled(self)

    def recast(self, old_bits: int, new_bits: int) -> Swizzle:
        return _cute_ffi_api.swizzle_recast(self, int(old_bits), int(new_bits))

    def to_swizzle_mode(self) -> SwizzleMode:
        return _cute_ffi_api.swizzle_to_swizzle_mode(self)


@tvm_ffi.register_object("tl.cute.IntTuple")
class IntTuple(Node, Scriptable):
    def __add__(self, other: IntTupleLike) -> IntTuple:
        return _cute_ffi_api.int_tuple_add(self, from_python(other))

    def __radd__(self, other: IntTupleLike) -> IntTuple:
        return _cute_ffi_api.int_tuple_add(from_python(other), self)

    def __mul__(self, other: IntTupleLike) -> IntTuple:
        return _cute_ffi_api.int_tuple_mul(self, from_python(other))

    def __rmul__(self, other: IntTupleLike) -> IntTuple:
        return _cute_ffi_api.int_tuple_mul(from_python(other), self)


@tvm_ffi.register_object("tl.cute.IntTupleConst")
class IntTupleConst(IntTuple):
    value: int


@tvm_ffi.register_object("tl.cute.IntTuplePrimExpr")
class IntTuplePrimExpr(IntTuple):
    value: object


@tvm_ffi.register_object("tl.cute.IntTupleScaledBasis")
class IntTupleScaledBasis(IntTuple):
    value: IntTuple
    basis: list


@tvm_ffi.register_object("tl.cute.IntTupleTuple")
class IntTupleTuple(IntTuple):
    fields: list


class ScaledBasis:
    """A ScaledBasis wrapper."""

    def __init__(self, value: PyIntTuple, mode: ModeLike):
        self._value = value
        self._mode = (mode,) if isinstance(mode, int) else tuple(int(m) for m in mode)

    @property
    def value(self) -> PyIntTuple:
        return self._value

    @property
    def mode(self) -> tuple:
        return self._mode

    def __repr__(self) -> str:
        return "@".join([str(self._value), *(str(m) for m in reversed(self._mode))])


def E(mode: ModeLike) -> ScaledBasis:
    return ScaledBasis(1, mode)


def product(shape: IntTupleLike) -> PyIntTuple:
    return to_python(_cute_ffi_api.product(from_python(shape)))


def flatten_to_tuple(value: IntTupleLike) -> tuple:
    if isinstance(value, IntTuple):
        value = to_python(value)
    if not isinstance(value, (tuple, list)):
        return (value,)
    return tuple(chain.from_iterable(flatten_to_tuple(x) for x in value))


@tvm_ffi.register_object("tl.cute.Layout")
class Layout(Node, Scriptable):
    @property
    def shape(self):
        return to_python(_cute_ffi_api.layout_shape(self))

    @property
    def stride(self):
        return to_python(_cute_ffi_api.layout_stride(self))

    def __getitem__(self, idx: int) -> Layout:
        return _cute_ffi_api.layout_get(self, int(idx))

    def __call__(self, coord: IntTupleLike):
        return to_python(_cute_ffi_api.layout_eval(self, from_python(coord)))

    def with_shape(self, shape: IntTupleLike) -> Layout:
        return _cute_ffi_api.with_shape(self, from_python(shape))

    @staticmethod
    def from_tilelang(layout) -> Layout | None:
        return _cute_ffi_api.layout_from_tilelang(layout)


def rank(layout: Layout) -> int:
    return int(_cute_ffi_api.layout_rank(layout))


def flatten(layout: Layout) -> Layout:
    return _cute_ffi_api.flatten(layout)


def size(layout: Layout | ComposedLayout) -> PyIntTuple:
    if isinstance(layout, ComposedLayout):
        layout = layout.layout
    return to_python(_cute_ffi_api.layout_size(layout))


def coalesce(layout: Layout, max_extent: int | None = None) -> Layout:
    if max_extent is None:
        return _cute_ffi_api.coalesce(layout)
    return _cute_ffi_api.coalesce_max(layout, int(max_extent))


def right_inverse(layout: Layout) -> Layout:
    return _cute_ffi_api.right_inverse(layout)


def composition(lhs: Layout, rhs: Layout) -> Layout:
    return _cute_ffi_api.composition(lhs, rhs)


def filter(layout: Layout) -> Layout:  # noqa: A001 - mirrors CuTe `filter`
    return _cute_ffi_api.filter(layout)


def congruent(a: IntTupleLike, b: IntTupleLike) -> bool:
    return bool(_cute_ffi_api.congruent(from_python(a), from_python(b)))


def cosize(layout: Layout) -> int:
    return to_python(_cute_ffi_api.cosize(layout))


def complement(layout: Layout, cotarget: int) -> Layout:
    return _cute_ffi_api.complement(layout, int(cotarget))


def logical_divide(layout: Layout, tiler: Layout) -> Layout:
    return _cute_ffi_api.logical_divide(layout, tiler)


def make_layout(shape: IntTupleLike, stride=None) -> Layout:
    # Concat: make_layout([layout0, layout1, ...])
    if stride is None and isinstance(shape, (tuple, list)) and shape and all(isinstance(x, Layout) for x in shape):
        return _cute_ffi_api.make_layout_concat(list(shape))
    if stride is None:
        return _cute_ffi_api.make_column_major_layout(from_python(shape))
    return _cute_ffi_api.make_layout(from_python(shape), from_python(stride))


def make_column_major_layout(shape: IntTupleLike) -> Layout:
    return _cute_ffi_api.make_column_major_layout(from_python(shape))


def make_row_major_layout(shape: IntTupleLike) -> Layout:
    return _cute_ffi_api.make_row_major_layout(from_python(shape))


def make_identity_layout(shape: IntTupleLike) -> Layout:
    return _cute_ffi_api.make_identity_layout(from_python(shape))


@tvm_ffi.register_object("tl.cute.ComposedLayout")
class ComposedLayout(Node, Scriptable):
    swizzle: Swizzle
    offset: int
    layout: Layout

    def recast(self, old_bits: int, new_bits: int) -> ComposedLayout:
        return _cute_ffi_api.composed_layout_recast(self, int(old_bits), int(new_bits))

    @staticmethod
    def from_tilelang(layout, buffer: BufferLikeType = None) -> ComposedLayout | None:
        mode = _cute_ffi_api.composed_layout_from_tilelang(layout)
        if buffer is None or mode is None:
            return mode
        from tilelang.layout.swizzle import _get_buffer_info

        _, _, dtype = _get_buffer_info(buffer)
        return mode.recast(int(tvm.DataType(dtype).bits), 8)


def restrict(layout: Layout, region: Sequence[Range]) -> tuple[PyIntTuple, Layout]:
    offset, sublayout = _cute_ffi_api.restrict(layout, list(region))
    return to_python(offset), sublayout
