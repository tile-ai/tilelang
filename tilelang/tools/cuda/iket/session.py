"""IKET compilation-session and CUDA callback lifecycle."""

from __future__ import annotations

import os
import threading
from typing import Any

import tvm_ffi

from tilelang.env import CacheState, disable_cache, enable_cache

from .cli import _restore_output_dir, _snapshot_output_dir, set_output_dir
from .codegen import inject_iket_cuda
from .frontend import reset


_CUDA_POSTPROC = "tilelang_callback_cuda_postproc"
_state_lock = threading.RLock()
_enable_depth = 0
_previous_cuda_postproc = None
_runtime_payloads_enabled = False


def enable_runtime_payloads() -> None:
    """Enable runtime payload metadata and records for subsequent compiles."""
    global _runtime_payloads_enabled
    with _state_lock:
        _runtime_payloads_enabled = True


def disable_runtime_payloads() -> None:
    """Disable payload records while keeping event instrumentation enabled."""
    global _runtime_payloads_enabled
    with _state_lock:
        _runtime_payloads_enabled = False


def runtime_payloads_enabled() -> bool:
    """Return whether runtime payload records are enabled."""
    with _state_lock:
        return _runtime_payloads_enabled


def _cuda_postproc(code: str, target: Any) -> str:
    return inject_iket_cuda(code, target, runtime_payloads=runtime_payloads_enabled())


def enable(*, override: bool = True) -> None:
    """Enable IKET CUDA source post-processing for subsequent compiles."""
    global _enable_depth, _previous_cuda_postproc
    with _state_lock:
        if _enable_depth:
            _enable_depth += 1
            return

        previous = tvm_ffi.get_global_func(_CUDA_POSTPROC, allow_missing=True)
        tvm_ffi.register_global_func(_CUDA_POSTPROC, f=_cuda_postproc, override=override)
        _previous_cuda_postproc = previous
        _enable_depth = 1


def disable(*, restore: bool = True) -> None:
    """Release one IKET enable scope and restore the prior callback at depth zero."""
    global _enable_depth, _previous_cuda_postproc
    with _state_lock:
        if not _enable_depth:
            return
        _enable_depth -= 1
        if _enable_depth:
            return

        previous = _previous_cuda_postproc
        _previous_cuda_postproc = None
        if restore and previous is not None:
            tvm_ffi.register_global_func(_CUDA_POSTPROC, f=previous, override=True)
        else:
            tvm_ffi.remove_global_func(_CUDA_POSTPROC)


def is_enabled() -> bool:
    """Return whether an IKET callback scope is active."""
    with _state_lock:
        return _enable_depth > 0


class _Session:
    def __init__(
        self,
        *,
        reset_events: bool = True,
        override: bool = True,
        disable_on_exit: bool = True,
        output_dir: str | os.PathLike[str] | None = None,
        runtime_payloads: bool | None = None,
        disable_cache: bool = True,
    ):
        self.reset_events = reset_events
        self.override = override
        self.disable_on_exit = disable_on_exit
        self.output_dir = output_dir
        self.runtime_payloads = runtime_payloads
        self.disable_cache = disable_cache
        self._previous_output_dir = None
        self._previous_env_output_dir = None
        self._previous_runtime_payloads = False
        self._previous_cache_enabled = False
        self._entered = False
        self._enabled_by_session = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("An IKET session object cannot be entered more than once at a time")
        self._previous_output_dir, self._previous_env_output_dir = _snapshot_output_dir()
        self._previous_runtime_payloads = runtime_payloads_enabled()
        self._previous_cache_enabled = CacheState.is_enabled()
        try:
            if self.output_dir is not None:
                set_output_dir(self.output_dir)
            if self.runtime_payloads is not None:
                if self.runtime_payloads:
                    enable_runtime_payloads()
                else:
                    disable_runtime_payloads()
            if self.disable_cache:
                disable_cache()
            enable(override=self.override)
            self._enabled_by_session = True
            if self.reset_events:
                reset()
        except Exception:
            try:
                if self._enabled_by_session:
                    disable(restore=True)
            finally:
                self._enabled_by_session = False
                self._restore_state()
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.disable_on_exit and self._enabled_by_session:
                disable(restore=True)
        finally:
            self._enabled_by_session = False
            self._restore_state()
            self._entered = False
        return False

    def _restore_state(self) -> None:
        _restore_output_dir(self._previous_output_dir, self._previous_env_output_dir)
        if self._previous_runtime_payloads:
            enable_runtime_payloads()
        else:
            disable_runtime_payloads()
        if self.disable_cache:
            if self._previous_cache_enabled:
                enable_cache()
            else:
                disable_cache()


def session(
    *,
    reset_events: bool = True,
    override: bool = True,
    disable_on_exit: bool = True,
    output_dir: str | os.PathLike[str] | None = None,
    runtime_payloads: bool | None = None,
    disable_cache: bool = True,
) -> _Session:
    """Configure IKET around kernel construction and compilation."""
    return _Session(
        reset_events=reset_events,
        override=override,
        disable_on_exit=disable_on_exit,
        output_dir=output_dir,
        runtime_payloads=runtime_payloads,
        disable_cache=disable_cache,
    )
