"""Metal-specific low-level TIR script operators."""

from __future__ import annotations

from tvm_tirx.backend.metal.script import MetalNamespace

_parser = MetalNamespace()

from tilelang.language.tir.exports import METAL_ONLY_TIR_EXPORTS

__all__ = tuple(sorted(METAL_ONLY_TIR_EXPORTS))


def __getattr__(name: str):
    if name in __all__:
        return getattr(_parser, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
