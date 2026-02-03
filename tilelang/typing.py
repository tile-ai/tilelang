"""Type annotations for TileLang."""

# Python 3.9 compatibility
try:
    from typing import TypeAlias
except ImportError:  # Python < 3.10
    from typing_extensions import TypeAlias

from tvm import tir
from tvm.tir import BufferLoad, BufferRegion
from tilelang.dtypes import AnyDType
from typing import Union

# Barrier can only be a Buffer, a BufferLoad
BarrierType: TypeAlias = tir.Buffer | BufferLoad

# BufferLikeType can be a Buffer, a BufferLoad, a BufferRegion
BufferLikeType: TypeAlias = tir.Buffer | BufferLoad | BufferRegion

# This is for Python 3.9 compatibility.
# In Python 3.9, we can only use isinstance(a, (TypeA, TypeB, ...)) instead of isinstance(a, TypeA | TypeB | ...))
BufferLikeTypeTuple = (tir.Buffer, BufferLoad, BufferRegion)

# Difference between "AnyDType" and "DType":
# - AnyDType is a union of all possible types that can represent a data type, including torch.dtype
# - DType is a more specific type alias that represents a data type in the context of TileLang, and must be
#   adapted to string.
DType: TypeAlias = Union[AnyDType, str]
