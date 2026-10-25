"""Native scalar domains and tensor inputs for TileLang layouts."""

from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang import _ffi_api
from tilelang._compat.tensor import TensorView

BufferLikeType = TensorView | ir.Var | ir.TensorLoad | ir.TensorRegion


@tvm_ffi.register_object("tl.LayoutAxis")
class LayoutAxis(tvm_ffi.Object):
    dom: ir.Range
    var: ir.Var

    def __init__(self, domain: ir.Range, variable: ir.Var):
        self.__init_handle_by_constructor__(_ffi_api.LayoutAxis, domain, variable)


def as_tensor_view(value: BufferLikeType) -> TensorView:
    if isinstance(value, TensorView):
        return value
    if isinstance(value, (ir.TensorLoad, ir.TensorRegion)):
        value = value.source
    if tirx.is_tensor_var(value):
        return TensorView(value)
    raise TypeError(f"Expected a native tensor, tensor access, or TensorView, got {type(value)}")
