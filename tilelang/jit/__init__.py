"""Lazy JIT entry points; importing exception types never loads the compiler path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from collections.abc import Callable, Iterable

if TYPE_CHECKING:
    from ._entry import ExecutionBackend, JITImpl, JITKernel, PrimFunc, TargetLike, _KP, _P, _T

from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
from threading import RLock

_ENTRY_NAMES = frozenset(
    (
        "logger",
        "_P",
        "_KP",
        "_T",
        "_Ret",
        "TargetLike",
        "_CallFormKey",
        "_CALL_FORM_CACHE_MISS",
        "_CallFormCache",
        "compile",
        "par_compile",
        "JITImpl",
        "ExecutionBackend",
        "jit",
        "JITKernel",
        "PrimFunc",
        "JITFunc",
        "Target",
    )
)
__all__ = ("jit", "compile", "par_compile", "JITImpl", "JITKernel")
_entry_loaded = False
_entry_loading = False
_entry_lock = RLock()


def __getattr__(name):
    if name == "JITKernel":
        from .kernel import JITKernel

        globals()[name] = JITKernel
        return JITKernel
    if name not in _ENTRY_NAMES or _entry_loading:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_entry()
    if name not in globals():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals()[name]


def _load_entry():
    global _entry_loaded, _entry_loading
    with _entry_lock:
        if _entry_loaded:
            return
        _entry_loading = True
        before = dict(globals())
        try:
            # Keep function/class __module__ and package-global monkeypatching
            # semantics from the original eager initializer. The implementation
            # is executed once in this package's actual namespace.
            loader = SourceFileLoader(__name__, str(Path(__file__).with_name("_entry.py")))
            loader.exec_module(sys.modules[__name__])
        except BaseException:
            for key in set(globals()) - before.keys():
                globals().pop(key)
            globals().update(before)
            raise
        else:
            _entry_loaded = True
        finally:
            _entry_loading = False


def __dir__():
    return sorted(set(globals()) | set(__all__))


def compile(
    func: PrimFunc[_KP, _T] = None,
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "pto"] | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel[_KP, _T]:
    """Load the implementation on first use, keeping the public signature."""
    _load_entry()
    return globals()["compile"](
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def par_compile(
    funcs: Iterable[PrimFunc[_KP, _T]],
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "pto"] | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    num_workers: int | None = None,
    ignore_error: bool = False,
) -> list[JITKernel[_KP, _T]]:
    """Load the implementation on first use, keeping the public signature."""
    _load_entry()
    return globals()["par_compile"](
        funcs=funcs,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
        num_workers=num_workers,
        ignore_error=ignore_error,
    )


def jit(
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    out_idx: list[int] | int | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable[[Callable[_P, _T]], JITImpl[_KP, _KP, _T, _T]]:
    """Load the implementation on first use, keeping the public signature."""
    _load_entry()
    return globals()["jit"](
        func=func,
        out_idx=out_idx,
        target=target,
        target_host=target_host,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        debug_root_path=debug_root_path,
        compile_flags=compile_flags,
    )
