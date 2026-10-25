"""Backend-neutral view of the upstream TIR script parser."""

from __future__ import annotations

import tvm_tirx.tirx.script as _parser
from tvm_tirx.ir import prim
from tilelang._compat.builder import bind

from .exports import BACKEND_ONLY_TIR_EXPORTS

__all__ = tuple(name for name in _parser.__all__ if name not in BACKEND_ONLY_TIR_EXPORTS) + ("align_up", "bind")


def align_up(value, alignment):
    return prim.ceildiv(value, alignment) * alignment


def __getattr__(name: str):
    if name in __all__:
        return getattr(_parser, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
