"""Tile operators exposed without importing backend implementations eagerly."""

from importlib import import_module

import tvm_ffi

from .base import GemmWarpPolicy

_EXPORTS = {"Gemm": "gemm", "GemmBlockScaled": "gemm_blockscaled", "GemmSP": "gemm_sp"}
__all__ = ("GemmWarpPolicy", *_EXPORTS)


def _register_default_callback(name):
    """Install a builtin callback without replacing an earlier user override."""

    def register(function):
        if tvm_ffi.get_global_func(name, allow_missing=True) is None:
            try:
                tvm_ffi.register_global_func(name, function)
            except RuntimeError as error:
                # A concurrent caller can register between lookup and insert.
                # FFI's non-overriding insert keeps that callback authoritative.
                if str(error) != f"Global Function `{name}` is already registered":
                    raise
        return function

    return register


def _register_builtin_callbacks(*backends):
    """Load native Pass callbacks when the Pass is requested by its caller.

    A generic Pass can see functions with different target attributes, so it
    needs every builtin op registry. Backend-specific Passes request only their
    own registry. Default registration preserves callbacks installed before the
    first import; import caching also preserves overrides on later calls.
    Device toolchains and unrelated backend pipelines are not initialized here.
    """
    for module in _EXPORTS.values():
        import_module(f".{module}", __name__)
    for backend in backends or ("cpu", "cuda", "rocm", "metal", "ascend"):
        import_module(f"tilelang.{backend}.op")


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
