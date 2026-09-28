"""Process-wide compilation phase enforcement for strict JIT execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
import threading
from typing import Any, TypeVar, cast

from tilelang.env import env


_F = TypeVar("_F", bound=Callable[..., Any])
_COMPILING = "compiling"
_SEALING = "sealing"
_EXECUTING = "executing"


class _CompilePhaseCoordinator:
    """Coordinate one irreversible compile-to-execute transition per process."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._local = threading.local()
        self._state = _COMPILING
        self._active_compilations = 0

    @contextmanager
    def compilation(self) -> Iterator[None]:
        """Register compilation work or reject it after execution has begun."""
        if not env.is_explicit_compile_required():
            yield
            return

        depth = getattr(self._local, "compilation_depth", 0)
        if depth:
            self._local.compilation_depth = depth + 1
            try:
                yield
            finally:
                self._local.compilation_depth = depth
            return

        with self._condition:
            if self._state != _COMPILING:
                raise RuntimeError(
                    "TileLang compilation is sealed while TILELANG_REQUIRE_EXPLICIT_COMPILE=1. "
                    "Compile every kernel before the first launch or before calling "
                    "tilelang.seal_compilation(), or start a new Python process for a new "
                    "compilation phase."
                )
            self._active_compilations += 1
            self._local.compilation_depth = 1

        try:
            yield
        finally:
            self._local.compilation_depth = 0
            with self._condition:
                self._active_compilations -= 1
                if self._active_compilations == 0:
                    self._condition.notify_all()

    def seal(self) -> None:
        """Wait for active compiles, then irreversibly enter execution phase."""
        if not env.is_explicit_compile_required():
            return
        if getattr(self._local, "compilation_depth", 0):
            raise RuntimeError(
                "Cannot launch a TileLang kernel from a thread that is still compiling while "
                "TILELANG_REQUIRE_EXPLICIT_COMPILE=1. Finish compilation before execution."
            )

        with self._condition:
            if self._state == _EXECUTING:
                return
            if self._state == _COMPILING:
                self._state = _SEALING

            while self._state == _SEALING and self._active_compilations:
                self._condition.wait()

            if self._state == _SEALING:
                self._state = _EXECUTING
                self._condition.notify_all()

    def is_sealed(self) -> bool:
        """Return whether this process has entered its execution phase."""
        with self._condition:
            return self._state == _EXECUTING

    def reset_for_testing(self) -> None:
        """Reset global state for an isolated test; never expose this as public API."""
        with self._condition:
            if self._active_compilations:
                raise RuntimeError("Cannot reset TileLang compilation phase while compilation is active.")
            self._state = _COMPILING
            self._local.compilation_depth = 0
            self._condition.notify_all()


_COORDINATOR = _CompilePhaseCoordinator()


@contextmanager
def compilation_scope() -> Iterator[None]:
    """Guard compilation work under the process-wide strict-mode phase."""
    with _COORDINATOR.compilation():
        yield


def compilation_guard(func: _F) -> _F:
    """Decorate a compilation entry point with the strict-mode phase guard."""

    @wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with compilation_scope():
            return func(*args, **kwargs)

    return cast(_F, guarded)


def guard_kernel_launch(func: _F) -> _F:
    """Wrap a kernel callable so its first launch seals strict-mode compilation."""
    if not env.is_explicit_compile_required():
        return func

    @wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        seal_compilation()
        return func(*args, **kwargs)

    return cast(_F, guarded)


def seal_compilation() -> None:
    """Finish the strict compilation phase before launching kernels.

    With ``TILELANG_REQUIRE_EXPLICIT_COMPILE=1``, this waits for active
    compilation calls and permanently rejects new compilation in the current
    process. Kernel launch callables invoke it automatically; harnesses may
    call it explicitly to establish the resource-allocation boundary. It is a
    no-op when explicit compilation mode is disabled.
    """
    _COORDINATOR.seal()


def is_compilation_sealed() -> bool:
    """Return whether strict compilation is sealed in the current process."""
    return _COORDINATOR.is_sealed()


def _reset_compilation_phase_for_testing() -> None:
    """Reset the process-wide phase for test isolation."""
    _COORDINATOR.reset_for_testing()
