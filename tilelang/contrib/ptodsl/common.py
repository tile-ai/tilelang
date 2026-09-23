"""Common PTODSL compatibility helpers used by generated TileLang kernels."""

from __future__ import annotations

from ptodsl import pto
from ptoas.mlir.ir import IntegerType
from ptodsl._ops import (
    _coerce_i1,
    _coerce_i16,
    _coerce_i32,
    _coerce_i64,
)
from ptodsl._scalar_adaptation import (
    coerce_runtime_integer_value as _coerce_runtime_integer_value,
)
from ptodsl._surface_values import (
    unwrap_surface_value as _unwrap_surface_value,
)
from ptodsl._surface_values import (
    wrap_surface_value as _wrap_surface_value,
)


def coerce_i1(value, *, context):
    """Coerce a scalar value to the PTODSL i1 representation."""
    return _coerce_i1(value, context=context)


def coerce_i8(value, *, context):
    """Coerce a scalar value to the PTODSL i8 representation."""
    return coerce_runtime_integer_value(unwrap_surface_value(value), pto.i8.resolve(), context=context)


def coerce_i16(value, *, context):
    """Coerce a scalar value to the PTODSL i16 representation."""
    return _coerce_i16(value, context=context)


def coerce_i32(value, *, context):
    """Coerce a scalar value to the PTODSL i32 representation."""
    return _coerce_i32(value, context=context)


def coerce_i64(value, *, context):
    """Coerce a scalar value to the PTODSL i64 representation."""
    return _coerce_i64(value, context=context)


def coerce_runtime_integer_value(value, target_type, *, context):
    """Coerce a runtime integer to the requested PTODSL scalar type."""
    return _coerce_runtime_integer_value(value, target_type, context=context)


def unwrap_surface_value(value):
    """Return the underlying IR value for a PTODSL surface value."""
    return _unwrap_surface_value(value)


def wrap_surface_value(value, **kwargs):
    """Wrap an IR value in PTODSL's public surface representation."""
    return _wrap_surface_value(value, **kwargs)


def as_logical_bool(value):
    """Normalize a scalar predicate to i1, including byte-backed i8 bools."""
    raw_value = unwrap_surface_value(value)
    if hasattr(raw_value, "type") and IntegerType.isinstance(raw_value.type) and IntegerType(raw_value.type).width == 1:
        return value
    return value != 0


def logical_not(value):
    """Negate a scalar i1 or byte-backed i8 predicate without Python truthiness."""
    return value == 0


def if_then_else(condition, true_fn, false_fn):
    """Trace a PTODSL conditional expression with lazily evaluated branches."""
    with pto.if_(condition) as branch:
        with branch.then_:
            branch.assign(value=true_fn())
        with branch.else_:
            branch.assign(value=false_fn())
    return branch.value
