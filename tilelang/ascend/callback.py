"""Ascend dialect code-generation callbacks.

The Ascend runtime-module build (src/ascend/codegen/rt_mod_ascend.cc) looks up
the ``tilelang_callback_ascend_postproc`` global before emitting the final
AscendC source; registering a function here lets users rewrite the generated
code the same way the CUDA/HIP postproc callbacks do for their backends.
"""

from __future__ import annotations

from collections.abc import Callable

import tvm_ffi
from tvm.target import Target

__all__ = ["register_ascend_postproc", "register_ascend_postproc_callback"]


def register_ascend_postproc(func: Callable[[str, Target], str], override: bool = True):
    """Register a post-processing function for Ascend-family code generation.

    The callback receives the generated AscendC source for an Ascend target.

    Args:
        func: A callable that takes generated code (str) and target (Target) as input,
              and returns the processed code (str).
        override: Whether to override existing registered function. Defaults to True.
    """
    tvm_ffi.register_global_func("tilelang_callback_ascend_postproc", f=func, override=override)


def register_ascend_postproc_callback(func: Callable | bool = None, override: bool = True):
    """Decorator for registering an Ascend-family post-processing callback.

    The callback receives the generated AscendC source for an Ascend target.

    Can be used with or without parentheses:
        @register_ascend_postproc_callback
        def func(code, target): ...

        @register_ascend_postproc_callback()
        def func(code, target): ...

        @register_ascend_postproc_callback(override=False)
        def func(code, target): ...

    Args:
        func: The function to be decorated or a boolean override flag
        override: Whether to override existing registered function. Defaults to True.
    """
    if callable(func):
        register_ascend_postproc(func, override)
        return func

    if func is None or isinstance(func, bool):
        _override = func if isinstance(func, bool) else override

        def _register(fn: Callable[[str, Target], str]):
            register_ascend_postproc(fn, _override)
            return fn

        return _register

    raise TypeError("Invalid decorator usage")
