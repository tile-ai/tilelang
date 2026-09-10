from types import SimpleNamespace

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.autotuner.tuner import AutoTuneImpl
from tilelang.jit.compile_phase import _reset_compilation_phase_for_testing


def _lazy_kernel_factory(size: int, block: int = 64):
    @T.prim_func
    def kernel():
        T.evaluate(size + block)

    return kernel


def _eager_copy(A, B, block: int = 8):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]

    with T.Kernel(T.ceildiv(M, block), T.ceildiv(N, block), threads=128):
        T.evaluate(0)


class _FakeKernel:
    def __init__(self):
        self.prepare_count = 0
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "executed"

    def prepare_for_execution(self):
        self.prepare_count += 1
        return self


class _FakeTuner:
    def __init__(self, kernel):
        self.kernel = kernel
        self.run_count = 0
        self.kernel_parameters = None

    def set_kernel_parameters(self, *args):
        self.kernel_parameters = args

    def run(self):
        self.run_count += 1
        self.kernel.prepare_for_execution()
        # AutoTuner.run benchmarks after candidate preparation.
        tilelang.seal_compilation()
        return SimpleNamespace(kernel=self.kernel, config={"block": 32})


@pytest.fixture(autouse=True)
def explicit_compile_environment(monkeypatch):
    _reset_compilation_phase_for_testing()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    yield
    _reset_compilation_phase_for_testing()


def test_autotune_requires_explicit_compile_before_lookup(monkeypatch):
    impl = AutoTuneImpl(jit_impl=tilelang.jit(_lazy_kernel_factory), configs=[{"block": 32}])
    kernel = _FakeKernel()
    tuner = _FakeTuner(kernel)
    monkeypatch.setattr(impl, "get_tunner", lambda: tuner)

    with pytest.raises(RuntimeError, match="No explicitly compiled specialization"):
        impl(8)
    assert tuner.run_count == 0

    assert impl.compile(8) is kernel
    assert kernel.prepare_count == 1
    assert tuner.run_count == 1
    assert impl(8) is kernel
    assert tuner.run_count == 1

    with pytest.raises(RuntimeError, match="No explicitly compiled specialization"):
        impl(16)
    assert tuner.run_count == 1
    with pytest.raises(RuntimeError, match="compilation is sealed"):
        impl.compile(16)


def test_autotune_candidate_compilation_is_explicit_in_strict_mode(monkeypatch):
    jit_impl = tilelang.jit(_lazy_kernel_factory)
    impl = AutoTuneImpl(jit_impl=jit_impl, configs=[{"block": 32}])
    sentinel = object()
    calls = []

    def explicit_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(jit_impl, "compile", explicit_compile)
    compile_candidate = impl._make_jit_compile_func("lazy", (8,), {})

    assert compile_candidate(block=32) is sentinel
    assert calls == [((8,), {"block": 32})]


def test_eager_autotune_compile_authorizes_matching_tensor_execution(monkeypatch):
    impl = AutoTuneImpl(jit_impl=tilelang.jit(_eager_copy), configs=[{"block": 8}])
    kernel = _FakeKernel()
    tuner = _FakeTuner(kernel)
    monkeypatch.setattr(impl, "get_tunner", lambda: tuner)
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    assert impl.compile(a, b) is kernel
    assert impl(torch.empty_like(a), torch.empty_like(b)) == "executed"
    assert len(kernel.calls) == 1
    assert tuner.run_count == 1

    with pytest.raises(RuntimeError, match="No explicitly compiled specialization"):
        impl(torch.empty(16, 8), torch.empty(16, 8))
