"""Type annotations for TileLang."""

# NOTE(chaofan): We should name it "_typing.py" to avoid module shadowing with standard library "typing"
# NOTE: In python 3.9, `from __future__ import annotations` does not for value expression, e.g. to define type alias

# Python 3.9 compatibility
try:
    from typing import TypeAlias
except ImportError:  # Python < 3.10
    from typing_extensions import TypeAlias

from typing import Union

from tvm import ir
from tvm import tir

from tvm.tir import BufferLoad, BufferRegion
from tilelang.dtypes import dtype

# Barrier can only be a Buffer, a BufferLoad
BarrierType: TypeAlias = Union[tir.Buffer, BufferLoad]

# BufferLikeType can be a Buffer, a BufferLoad, a BufferRegion
BufferLikeType: TypeAlias = Union[tir.Buffer, BufferLoad, BufferRegion]

# This is for Python 3.9 compatibility.
# In Python 3.9, we can only use isinstance(a, (TypeA, TypeB, ...)) instead of isinstance(a, TypeA | TypeB | ...))
BufferLikeTypeTuple = (tir.Buffer, BufferLoad, BufferRegion)

# Difference between "AnyDType" and "DType":
# - AnyDType is a union of all possible types that can represent a data type, including torch.dtype
# - DType is a more specific type alias that represents a data type in the context of TileLang, and must be
#   adapted to string.
DType: TypeAlias = Union[dtype, ir.Type, str, type]
ShapeType: TypeAlias = Union[list[Union[tir.PrimExpr, int]], tuple[Union[tir.PrimExpr, int], ...]]

# PrimExpr with adaptation to Python basic data types
# IntImm, FloatImm, Bool: IntImm, Integer: IntImm
PyPrimExpr: TypeAlias = Union[tir.PrimExpr, int, float, bool]
