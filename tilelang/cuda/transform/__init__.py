"""CUDA-specific transformation frontends."""

from tilelang.transform import _ffi_api


def LowerHopperIntrin():
    """LowerHopperIntrin"""
    if hasattr(_ffi_api, "LowerHopperIntrin"):
        return _ffi_api.LowerHopperIntrin()  # type: ignore
    return lambda f: f


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


__all__ = [
    "LowerHopperIntrin",
    "LowerL2Persistent",
    "PersistThreadblock",
]
