import importlib
import threading

import pytest

import tilelang
import tilelang.language as T
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.jit.compile_phase import (
    _reset_compilation_phase_for_testing,
    compilation_scope,
    is_compilation_sealed,
)
from tilelang.jit.kernel import JITKernel


jit_module = importlib.import_module("tilelang.jit")


@T.prim_func
def _empty_prim_func():
    T.evaluate(0)


class _RecordingAdapter(BaseKernelAdapter):
    def __init__(self):
        self.calls = []
        super().__init__(mod=None, params=[], result_idx=[])

    def _convert_torch_func(self):
        def launch(*args, **kwargs):
            self.calls.append((args, kwargs))
            return "launched"

        return launch


@pytest.fixture(autouse=True)
def isolated_compilation_phase(monkeypatch):
    _reset_compilation_phase_for_testing()
    monkeypatch.delenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", raising=False)
    yield
    _reset_compilation_phase_for_testing()


def test_default_mode_does_not_seal_compilation():
    adapter = _RecordingAdapter()

    tilelang.seal_compilation()
    assert adapter.func(1, named=2) == "launched"
    with compilation_scope():
        pass

    assert not is_compilation_sealed()
    assert adapter.calls == [((1,), {"named": 2})]


def test_first_kernel_launch_seals_compilation(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    adapter = _RecordingAdapter()

    with compilation_scope():
        pass
    assert adapter.func() == "launched"
    assert is_compilation_sealed()

    with pytest.raises(RuntimeError, match="compilation is sealed"), compilation_scope():
        pass

    # Once sealed, additional kernel launches remain valid.
    assert adapter() == "launched"


def test_explicit_seal_establishes_execution_boundary(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    adapter = _RecordingAdapter()

    tilelang.seal_compilation()

    assert is_compilation_sealed()
    assert adapter() == "launched"
    with pytest.raises(RuntimeError, match="start a new Python process"), compilation_scope():
        pass
    with pytest.raises(RuntimeError, match="compilation is sealed"):
        jit_module.par_compile([])


def test_compile_launch_compile_sequence_is_rejected(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    adapter = _RecordingAdapter()
    kernel = JITKernel.__new__(JITKernel)
    kernel.adapter = adapter
    kernel.torch_function = adapter.func
    cache_calls = []

    def fake_cached(**kwargs):
        cache_calls.append(kwargs)
        return kernel

    monkeypatch.setattr(jit_module, "cached", fake_cached)

    assert jit_module.compile(_empty_prim_func) is kernel
    assert kernel() == "launched"
    with pytest.raises(RuntimeError, match="compilation is sealed"):
        jit_module.compile(_empty_prim_func)

    assert len(cache_calls) == 1


def test_launch_waits_for_active_compilation(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    adapter = _RecordingAdapter()
    compile_started = threading.Event()
    release_compile = threading.Event()
    launch_started = threading.Event()
    launch_finished = threading.Event()
    errors = []

    def compile_worker():
        try:
            with compilation_scope():
                compile_started.set()
                if not release_compile.wait(timeout=2):
                    raise TimeoutError("test did not release compilation")
        except Exception as error:
            errors.append(error)

    def launch_worker():
        launch_started.set()
        try:
            adapter()
        except Exception as error:
            errors.append(error)
        finally:
            launch_finished.set()

    compile_thread = threading.Thread(target=compile_worker)
    launch_thread = threading.Thread(target=launch_worker)
    compile_thread.start()
    assert compile_started.wait(timeout=2)
    launch_thread.start()
    assert launch_started.wait(timeout=2)
    assert not launch_finished.wait(timeout=0.05)

    release_compile.set()
    compile_thread.join(timeout=2)
    launch_thread.join(timeout=2)

    assert not compile_thread.is_alive()
    assert not launch_thread.is_alive()
    assert errors == []
    assert is_compilation_sealed()


def test_same_thread_launch_during_compilation_fails_without_deadlock(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    adapter = _RecordingAdapter()

    with compilation_scope(), pytest.raises(RuntimeError, match="thread that is still compiling"):
        adapter()

    assert not is_compilation_sealed()
    assert adapter() == "launched"
