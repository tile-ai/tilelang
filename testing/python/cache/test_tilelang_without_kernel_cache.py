"""Tests for ``tilelang.testing.without_kernel_cache``."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import env
from tilelang.env import CacheState
from tilelang.jit import JITImpl


@pytest.fixture(autouse=True)
def _restore_cache_state():
    enabled = CacheState.is_enabled()
    try:
        yield
    finally:
        if enabled:
            CacheState.enable()
        else:
            CacheState.disable()


def _set_cache_state(enabled: bool) -> None:
    if enabled:
        CacheState.enable()
    else:
        CacheState.disable()


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_without_kernel_cache_restores_the_previous_state(initially_enabled):
    _set_cache_state(initially_enabled)

    with tilelang.testing.without_kernel_cache():
        assert not CacheState.is_enabled()

    assert CacheState.is_enabled() is initially_enabled


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_without_kernel_cache_restores_the_previous_state_after_the_body_reenables(initially_enabled):
    _set_cache_state(initially_enabled)

    with tilelang.testing.without_kernel_cache():
        CacheState.enable()

    assert CacheState.is_enabled() is initially_enabled


def test_without_kernel_cache_bypasses_the_jit_wrapper_caches(monkeypatch):
    """A jit call inside the block compiles instead of reusing an earlier kernel."""
    # The in-process caches are only consulted when the cache is enabled, so pin
    # both switches instead of inheriting them from the caller's environment.
    monkeypatch.setattr(vars(type(env))["TILELANG_DISABLE_CACHE"], "_forced_value", "0")
    _set_cache_state(True)
    compile_calls = []

    def fake_compile(self, *args, **kwargs):
        compile_calls.append(args)
        return object()

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    @tilelang.jit(out_idx=-1, target="c")
    def build(N):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(N):
                    C[i] = A[i] + B[i]

        return main

    first = build(128)
    assert build(128) is first
    assert len(compile_calls) == 1

    with tilelang.testing.without_kernel_cache():
        fresh = build(128)

    assert fresh is not first
    assert len(compile_calls) == 2

    assert build(128) is first
    assert build(N=128) is first
    assert len(compile_calls) == 2


if __name__ == "__main__":
    tilelang.testing.main()
