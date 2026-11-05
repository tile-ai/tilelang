"""Execution adapters loaded only for the selected runtime backend."""

from importlib import import_module

from .base import BaseKernelAdapter, CachedTextSource

_ADAPTERS = {
    "TVMFFIKernelAdapter": "tvm_ffi",
    "CythonKernelAdapter": "cython",
    "NVRTCKernelAdapter": "nvrtc",
    "MetalKernelAdapter": "torch",
    "CuTeDSLKernelAdapter": "cutedsl",
}
__all__ = ["BaseKernelAdapter", "CachedTextSource", *_ADAPTERS]


def __getattr__(name):
    if name not in _ADAPTERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_ADAPTERS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _ADAPTERS.keys())
