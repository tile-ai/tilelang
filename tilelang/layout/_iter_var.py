"""Coordinate domains used by the TileLang layout construction facade."""

import tvm_ffi
from tvm_tirx.ir import Range, Var
from tvm_tirx.ir._prim_expr_overload import ExprOp

from tilelang import _ffi_api


@tvm_ffi.register_object("tl.LayoutIterVar")
class LayoutIterVar(tvm_ffi.Object):
    """A native scalar variable and domain, independent of loop IR objects."""

    def __init__(self, dom: Range, var: Var):
        self.__init_handle_by_constructor__(_ffi_api.LayoutIterVar, dom, var)


class ReplicateCoordinate(ExprOp):
    """Keep the layout callback's ``rep.var``/arithmetic surface over a native Var.

    This short-lived Python facade is used only while calling ``forward_fn``.
    Every expression operator and FFI argument resolves to ``var``; this object
    is never embedded into native IR or retained in a layout payload.
    """

    def __init__(self, coordinate: LayoutIterVar):
        self.var = coordinate.var
        self.dom = coordinate.dom

    def expr_ty(self):
        return self.var.ty

    def __tvm_ffi_object__(self):
        return self.var
