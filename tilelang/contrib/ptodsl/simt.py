"""PTODSL tracing helpers for TileLang PTO SIMT codegen."""

from __future__ import annotations

import ptodsl._allreduce as _allreduce
from ptodsl import pto
from ptodsl._surface_values import unwrap_surface_value, wrap_surface_value
from ptodsl._types import _integer_signedness, _restore_integer_signedness, _strip_integer_signedness
from ptoas.mlir.dialects import llvm
from ptoas.mlir.ir import IntegerType


def _vector_lane(value, index):
    raw_index = unwrap_surface_value(pto.const(index, dtype=pto.i32))
    return wrap_surface_value(llvm.ExtractElementOp(unwrap_surface_value(value), raw_index).res)


def vectorize_unary_f32x2(op, value):
    return pto.Vec(
        pto.f32,
        2,
        init=(
            op(_vector_lane(value, 0)),
            op(_vector_lane(value, 1)),
        ),
    )


def vectorize_binary_f32x2(op, lhs, rhs):
    return pto.Vec(
        pto.f32,
        2,
        init=(
            op(_vector_lane(lhs, 0), _vector_lane(rhs, 0)),
            op(_vector_lane(lhs, 1), _vector_lane(rhs, 1)),
        ),
    )


def scalar_div(lhs, rhs):
    return lhs / rhs


def scalar_rsqrt(value):
    return 1.0 / pto.sqrt(value)


def _redux_integer_compat(op, value):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return op(value)
    signedness = _integer_signedness(raw_value.type)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = op(signless_value, signedness=signedness)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


def _redux_add(value):
    return _redux_integer_compat(pto.redux_add, value)


def _redux_max(value):
    return _redux_integer_compat(pto.redux_max, value)


def _redux_min(value):
    return _redux_integer_compat(pto.redux_min, value)


def _shuffle_bfly(value, offset):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return pto.shuffle_bfly(value, offset)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = pto.shuffle_bfly(signless_value, offset)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


# PTOAS v0.1.6 models integer signedness on PTODSL values, while its low-level
# redux and shuffle operations require signless LLVM-compatible i32 carriers.
# Keep the public allreduce implementation and adapt only those two boundaries.
_allreduce._REDUCER_REDUX.update(
    {
        "sum": _redux_add,
        "max": _redux_max,
        "min": _redux_min,
    }
)
_allreduce.shuffle_bfly = _shuffle_bfly


def simt_allreduce_sum(value, **kwargs):
    return _allreduce.simt_allreduce_sum(value, **kwargs)


def simt_allreduce_max(value, **kwargs):
    return _allreduce.simt_allreduce_max(value, **kwargs)


def simt_allreduce_min(value, **kwargs):
    return _allreduce.simt_allreduce_min(value, **kwargs)
