"""Compiler toolchains loaded only when their backend is requested."""

from importlib import import_module

_EXPORTS = {"compile_cuda": "nvcc", "compile_hip": "hipcc", "compile_ascend": "bisheng"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
