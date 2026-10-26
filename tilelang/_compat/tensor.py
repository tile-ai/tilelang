"""Explicit Buffer-to-Tensor boundary helpers.

``TensorView`` is a Python view, not an IR node. Only its native ``Var`` enters
IR. Allocation and pointer declaration return explicit Bind statements, so a
caller cannot accidentally lose the allocation's lifetime or backing pointer.
Address geometry belongs exclusively to TensorType.layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import Axis, Layout, S, TileLayout

from .memory_scope import fragment_scope, reducer_scope


def tensor_parameter_map(function):
    """Return tensor parameters keyed by their native parameter identity."""
    return {parameter: parameter for parameter in function.params if tirx.is_tensor_var(parameter)}


def tensor_strides(tensor):
    """Read a strided tensor contract without inventing strides for swizzled layouts."""
    from tilelang._compat import sym

    layout = tensor.ty.layout
    if not isinstance(layout, TileLayout) or layout.replica or len(layout.shard) != len(tensor.shape):
        raise ValueError(f"Tensor {tensor.name!r} does not have a rank-preserving strided layout")
    analyzer = sym.Analyzer()
    for extent, iterator in zip(tensor.shape, layout.shard):
        if iterator.axis != Axis.m or not analyzer.can_prove_equal(extent, iterator.extent):
            raise ValueError(f"Tensor {tensor.name!r} does not have a pure-memory strided layout")
    return [iterator.stride for iterator in layout.shard]


def tensor_element_offset(tensor):
    """Read the memory-axis origin from native tensor layout metadata."""
    layout = tensor.ty.layout
    if layout is None:
        raise ValueError(f"Tensor {tensor.name!r} has no defined storage layout")
    return layout.apply(*[0 for _ in tensor.shape], shape=tensor.shape).get("m", ir.const(0))


def tensor_type(
    shape: Sequence[int | ir.Expr],
    dtype,
    *,
    scope="global",
    strides: Sequence[int | ir.Expr] | None = None,
    elem_offset: int | ir.Expr = 0,
    data_alignment: int = 0,
    layout: Layout | None = None,
) -> tirx.TensorType:
    """Translate Buffer address geometry into a single native layout."""
    shape = (shape,) if isinstance(shape, (int, ir.Expr)) else tuple(shape)
    if scope == "":
        scope = "global"
    if scope == "local.fragment":
        scope = fragment_scope()
    elif scope == "local.reducer":
        scope = reducer_scope()
    if layout is not None:
        if strides is not None or not isinstance(elem_offset, int) or elem_offset != 0:
            raise ValueError("An explicit layout cannot be combined with strides or an element offset")
    else:
        if strides is not None:
            strides = tuple(strides)
            if len(strides) != len(shape):
                raise ValueError("strides and shape must have the same rank")
            specification = S[shape:strides]
        else:
            specification = S[shape]
        layout = TileLayout(specification + elem_offset @ Axis.m)
    return tirx.TensorType(shape, dtype, scope=scope, data_alignment=data_alignment, layout=layout)


@dataclass(frozen=True, eq=False)
class TensorView:
    """Identity-preserving view over a Var with a native TensorType."""

    tensor: ir.Var

    def __post_init__(self):
        if not tirx.is_tensor_var(self.tensor):
            raise TypeError("TensorView requires a native Var with TensorType")

    @classmethod
    def create(cls, name: str, shape: Sequence[int | ir.Expr], dtype, *, loc=ir.UNKNOWN_LOC, **geometry):
        return cls(ir.Var(name, tensor_type(shape, dtype, **geometry), loc=loc))

    @property
    def shape(self):
        return self.tensor.ty.shape

    @property
    def dtype(self):
        return self.tensor.ty.dtype.dtype

    @property
    def layout(self):
        return self.tensor.ty.layout

    @property
    def data(self):
        return tirx.tensor_data_ptr(self.tensor)

    def scope(self) -> str:
        return self.tensor.ty.scope.name

    def same_as(self, other) -> bool:
        return isinstance(other, TensorView) and self.tensor.same_as(other.tensor)

    def __eq__(self, other) -> bool:
        return self.same_as(other)

    def __hash__(self) -> int:
        return self.tensor.__chandle__()

    def _indices(self, indices):
        if isinstance(indices, (int, ir.Expr)):
            indices = [indices]
        indices = list(indices)
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, received {len(indices)}")
        return indices

    def load(self, indices, *, loc=ir.UNKNOWN_LOC) -> ir.TensorLoad:
        return ir.TensorLoad(self.tensor, self._indices(indices), loc=loc)

    def store(self, indices, value, *, loc=ir.UNKNOWN_LOC) -> ir.TensorStore:
        if not isinstance(value, ir.Expr):
            value = ir.const(value, self.tensor.ty.dtype)
        return ir.TensorStore(self.tensor, self._indices(indices), value, loc=loc)

    def region(self, ranges: Sequence[ir.Range], *, loc=ir.UNKNOWN_LOC) -> ir.TensorRegion:
        if len(ranges) != len(self.shape):
            raise ValueError("Tensor regions must have one range per dimension")
        return ir.TensorRegion(self.tensor, list(ranges), ir.TensorRegionType(), loc=loc)

    def pointer_at(self, indices) -> ir.Expr:
        return self.tensor.ptr_to(self._indices(indices))

    def allocate(self, *, loc=ir.UNKNOWN_LOC) -> ir.Bind:
        allocation = ir.Call("tirx.alloc_tensor", [], attrs={}, ty_args=[self.tensor.ty], loc=loc)
        return ir.Bind(self.tensor, allocation, loc=loc)

    def declare(self, data: ir.Expr, *, loc=ir.UNKNOWN_LOC) -> ir.Bind:
        if not isinstance(data, ir.Expr) or not isinstance(data.ty, ir.PtrType):
            raise TypeError("Tensor declarations require a typed pointer expression")
        expected = ir.PtrType(self.tensor.ty.dtype, self.tensor.ty.scope)
        if data.ty != expected:
            raise TypeError(f"Backing pointer has type {data.ty}, expected {expected}")
        declaration = ir.Call("tirx.decl_tensor", [data], ty_args=[self.tensor.ty], loc=loc)
        return ir.Bind(self.tensor, declaration, loc=loc)
