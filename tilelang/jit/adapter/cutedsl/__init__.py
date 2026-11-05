"""CuTeDSL Backend for TileLang.

This module provides runtime compilation support using NVIDIA's CuTeDSL API.
"""

__all__ = [
    "CuTeDSLKernelAdapter",
    "TLCuTeDSLSourceWrapper",
    "CuTeDSLLibraryGenerator",
    "check_cutedsl_available",
]

from .checks import check_cutedsl_available


def __getattr__(name):
    # Target discovery only needs the capability check. Keep compiler/runtime
    # construction lazy, including when loading an already compiled artifact.
    from importlib import import_module

    modules = {
        "CuTeDSLKernelAdapter": ".adapter",
        "TLCuTeDSLSourceWrapper": ".wrapper",
        "CuTeDSLLibraryGenerator": ".libgen",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
