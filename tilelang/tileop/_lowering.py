"""Closed native functions at the Python/C++ tile-operator boundary."""

from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir, tirx


def close_lowered_function(function: tirx.Function) -> tuple[tirx.Function, list[ir.Expr]]:
    """Make a tile-op implementation's lexical captures explicit.

    Backend generators use a parameterless ``@T.prim_func`` to describe an
    implementation in terms of the calling kernel's tensors and coordinates.
    Such a function is an intermediate construction result, not a callable
    compiler function. Its free variables include variables used only by a
    tensor's shape or layout, so inspecting load/store operands alone is
    insufficient.

    TensorLoad/Store use analysis reports their tensor sources; an additional
    expression traversal of those captures expands type/layout dependencies
    before the tensor that uses them. We retain that order
    in both the new signature and the parallel capture-argument array. C++ must
    renew the returned function's definitions, substitute the arguments using
    a binder-aware mutator, and inline its SeqStmt inside a lexical allocation
    scope. It must not simply take ``function.body`` from an open function.

    Source functions with explicit parameters are rejected: there is no call
    site mapping for them in this parameterless generator protocol.
    """
    if not isinstance(function, tirx.Function):
        raise TypeError("A tile-op lowerer must return a native tirx.Function")
    if function.attrs.get("declaration", False):
        raise ValueError("A tile-op lowerer cannot return a function declaration")
    if function.params:
        raise ValueError("Tile-op implementation generators must not declare explicit parameters")
    result_type = function.ret_type
    is_void = isinstance(result_type, ir.PrimType) and str(result_type.dtype) in ("", "void")
    is_unit = isinstance(result_type, ir.TupleType) and not result_type.fields
    if not (is_void or is_unit):
        raise TypeError("A tile-op implementation must return void")

    attribute_variables = []
    tvm_ffi.structural_walk(function.attrs, (ir.Var, attribute_variables.append), order="pre")
    if attribute_variables:
        raise ValueError("Tile-op IR references must be body operands, not opaque function attributes")

    sources = list(tirx.analysis.undefined_vars(function.body))
    captures = list(tirx.analysis.undefined_vars(ir.Tuple(sources)))
    closed = tirx.Function(captures, function.body, function.ret_type, function.attrs, function.loc)
    tirx.analysis.check_well_formed(closed)
    return closed, captures
