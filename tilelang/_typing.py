"""Type annotations and value predicates for TileLang."""

# NOTE: Future annotations do not defer the expressions used for type aliases.
from typing import TypeAlias

from tvm_tirx import ir

from tilelang.language._tensor_spec import TensorValue, is_buffer_like
from tilelang.dtypes import dtype

BarrierType: TypeAlias = TensorValue | ir.TensorLoad
BufferLikeType: TypeAlias = TensorValue | ir.TensorLoad | ir.TensorRegion


def is_buffer_value(value):
    """Accept source tensors, native tensor variables, loads, and regions.

    A native Var alone is insufficient: scalar Vars are expressions, not tensor
    storage. Keep this runtime check separate from the static annotation union.
    """
    return is_buffer_like(value) or isinstance(value, (ir.TensorLoad, ir.TensorRegion))


# AnyDType also accepts framework dtype objects; DType is the source annotation
# accepted at the DSL boundary and is normalized before native construction.
DType: TypeAlias = dtype | ir.Type | str | type
ShapeType: TypeAlias = list[ir.Expr | int] | tuple[ir.Expr | int, ...]
PyPrimExpr: TypeAlias = ir.Expr | int | float | bool
