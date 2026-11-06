"""Source reducer scopes and native thread-allreduce operands."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx import tirx as _tirx
from tvm_tirx.script.ir_builder import IRBuilder

from ._construction import as_expr, current_location


@dataclass(frozen=True)
class _ThreadReducer:
    """Construction policy carried explicitly by the native allreduce call."""

    combine: ir.LambdaExpr
    identity: tuple[ir.Expr, ...]


_thread_reducer_scope = ContextVar("tilelang_thread_reducer_scope", default=None)


class _ThreadReducerScope:
    def __init__(self, reducer):
        self.reducer = reducer
        self.token = None

    def __enter__(self):
        if self.token is not None:
            raise RuntimeError("A reduce_scope cannot be entered twice")
        self.token = _thread_reducer_scope.set((IRBuilder.current(), self.reducer))

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_reducer_scope.reset(self.token)
        self.token = None


def comm_reducer(combiner, identity):
    """Keep the TileLang reducer spelling on a typed native combining lambda.

    The callable receives all left-hand values followed by all right-hand values.
    Its result must have one element of the matching type per identity. This is
    construction metadata; the resulting IR uses ``tirx.gpu_thread_allreduce``
    with explicit combiner, identity, values, predicate, outputs and thread axes.
    """
    if not isinstance(identity, (list, tuple, tvm_ffi.Array)) or not identity:
        raise ValueError("comm_reducer requires a non-empty list of identity values")
    values = tuple(as_expr(value) for value in identity)
    types = [value.ty for value in values]
    if not all(isinstance(ty, ir.PrimType) for ty in types):
        raise TypeError("comm_reducer identity values must have primitive types")

    def combine(*args):
        result = as_expr(combiner(*args))
        return result if isinstance(result, ir.Tuple) else ir.Tuple([result])

    function = ir.LambdaExpr(types + types, combine, ret_type=ir.TupleType(types), loc=current_location())
    return _ThreadReducer(function, values)


def tvm_thread_allreduce(*args):
    """Bind the legacy argument order to the current typed reduction scope."""
    active = _thread_reducer_scope.get()
    if active is None or not active[0].same_as(IRBuilder.current()):
        raise ValueError("tvm_thread_allreduce requires an enclosing reduce_scope in the same function")
    if not args:
        raise ValueError("tvm_thread_allreduce requires a constant result count")
    count = as_expr(args[0])
    if not isinstance(count, ir.IntImm) or int(count) <= 0:
        raise ValueError("tvm_thread_allreduce requires a positive constant result count")
    count = int(count)
    reducer = active[1]
    if count != len(reducer.identity):
        raise ValueError("tvm_thread_allreduce result count does not match comm_reducer identities")
    # count, N values, predicate, N destinations, one or more thread axes.
    if len(args) < 2 * count + 3:
        raise ValueError("tvm_thread_allreduce requires values, predicate, destinations and thread axes")
    return _tirx.gpu_thread_allreduce(
        reducer.combine,
        reducer.identity,
        tuple(as_expr(value) for value in args[1 : 1 + count]),
        as_expr(args[1 + count]),
        tuple(as_expr(value) for value in args[2 + count : 2 + 2 * count]),
        tuple(as_expr(value) for value in args[2 + 2 * count :]),
        loc=current_location(),
    )
