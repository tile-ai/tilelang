"""Backend-neutral view of the upstream TIR script parser."""

from __future__ import annotations

import tvm.tirx.script.parser as _parser

from .exports import BACKEND_ONLY_TIR_EXPORTS

__all__ = tuple(name for name in _parser.__all__ if name not in BACKEND_ONLY_TIR_EXPORTS)


def __getattr__(name: str):
    if name in __all__:
        return getattr(_parser, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
