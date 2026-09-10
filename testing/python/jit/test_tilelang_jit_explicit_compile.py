import importlib
from types import SimpleNamespace

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter


jit_module = importlib.import_module("tilelang.jit")


def _lazy_kernel_factory(size: int):
    @T.prim_func
    def kernel():
        T.evaluate(size)

    return kernel


def _eager_copy(A, B, block: int = 8):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]

    with T.Kernel(T.ceildiv(M, block), T.ceildiv(N, block), threads=128) as (pid_m, pid_n):
        for i, j in T.Parallel(block, block):
            row = pid_m * block + i
            col = pid_n * block + j
            if row < M and col < N:
                B[row, col] = A[row, col]


@T.prim_func
def _empty_prim_func():
    T.evaluate(0)


class _FakeKernel:
    def __init__(self, result=None):
        self.result = result
        self.calls = []
        self.prepare_count = 0

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def prepare_for_execution(self):
        self.prepare_count += 1
        return self


@pytest.fixture
def fake_compile(monkeypatch):
    kernels = []

    def compile_prim_func(_func, **_kwargs):
        kernel = _FakeKernel(result=len(kernels))
        kernels.append(kernel)
        return kernel

    monkeypatch.setattr(jit_module, "compile", compile_prim_func)
    return kernels


@pytest.fixture(autouse=True)
def explicit_compile_environment(monkeypatch):
    # The explicit-specialization registry is intentionally independent of the
    # persistent/global kernel cache.
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.delenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", raising=False)


def test_default_mode_still_compiles_on_first_invocation(fake_compile):
    factory = tilelang.jit(_lazy_kernel_factory)

    assert factory(8) is fake_compile[0]
    assert factory(8) is fake_compile[0]
    assert len(fake_compile) == 1


def test_strict_mode_rejects_lazy_compile_on_invocation(monkeypatch, fake_compile):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    factory = tilelang.jit(_lazy_kernel_factory)

    with pytest.raises(RuntimeError, match=r"factory\.compile\(\.\.\.\)"):
        factory(8)

    assert fake_compile == []


def test_explicit_compile_registers_specialization_with_cache_disabled(monkeypatch, fake_compile):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    factory = tilelang.jit(_lazy_kernel_factory)

    compiled = factory.compile(8)

    assert compiled is fake_compile[0]
    assert factory(8) is compiled
    assert len(fake_compile) == 1
    with pytest.raises(RuntimeError, match="matching tensor shapes"):
        factory(16)


def test_implicit_compile_does_not_authorize_later_strict_invocation(monkeypatch, fake_compile):
    factory = tilelang.jit(_lazy_kernel_factory)
    implicit_kernel = factory(8)

    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    with pytest.raises(RuntimeError, match="No explicitly compiled specialization"):
        factory(8)

    explicit_kernel = factory.compile(8)
    assert explicit_kernel is not implicit_kernel
    assert factory(8) is explicit_kernel
    assert len(fake_compile) == 2


def test_eager_explicit_compile_then_invocation(monkeypatch, fake_compile):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    copy = tilelang.jit(_eager_copy)
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    compiled = copy.compile(a, b)

    assert copy(a, b) == 0
    assert compiled.calls == [((a, b), {})]
    with pytest.raises(RuntimeError, match="No explicitly compiled specialization"):
        copy(torch.empty(16, 8), torch.empty(16, 8))
    assert len(fake_compile) == 1


def test_shape_only_eager_compile_authorizes_equivalent_tensor_call(monkeypatch, fake_compile):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    copy = tilelang.jit(_eager_copy)
    compiled = copy.compile(M=8, N=8)
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    assert copy(a, b) == 0
    assert compiled.calls == [((a, b), {})]
    assert len(fake_compile) == 1


def test_parallel_compile_registers_every_specialization(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    factory = tilelang.jit(_lazy_kernel_factory)
    kernels = [_FakeKernel("eight"), _FakeKernel("sixteen")]

    def fake_par_compile(funcs, **_kwargs):
        assert len(list(funcs)) == 2
        return kernels

    monkeypatch.setattr(jit_module, "par_compile", fake_par_compile)

    assert factory.par_compile([(8,), (16,)]) == kernels
    assert factory(8) is kernels[0]
    assert factory(16) is kernels[1]
    with pytest.raises(RuntimeError, match="No explicitly compiled specialization"):
        factory(32)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_explicit_compile_environment_truthy_values(monkeypatch, value):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", value)
    assert tilelang.env.is_explicit_compile_required()


def test_top_level_compile_prepares_backend_in_strict_mode(monkeypatch):
    kernel = _FakeKernel()
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    monkeypatch.setattr(jit_module, "cached", lambda **_kwargs: kernel)

    assert jit_module.compile(_empty_prim_func) is kernel
    assert kernel.prepare_count == 1


def test_top_level_compile_preserves_lazy_backend_preparation_by_default(monkeypatch):
    kernel = _FakeKernel()
    monkeypatch.setattr(jit_module, "cached", lambda **_kwargs: kernel)

    assert jit_module.compile(_empty_prim_func) is kernel
    assert kernel.prepare_count == 0


def test_cutedsl_fresh_cubin_rejects_compile_before_execution_claim():
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.pymodule = SimpleNamespace(_cubin_needs_generation=True)

    with pytest.raises(RuntimeError, match="cannot prepare a fresh CuTeDSL cubin"):
        adapter.prepare_for_execution()


def test_cutedsl_precompiled_cubin_is_ready_for_execution():
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.pymodule = SimpleNamespace(_cubin_needs_generation=False)

    adapter.prepare_for_execution()


@tilelang.testing.requires_cuda
def test_real_eager_compile_then_execute(monkeypatch):
    monkeypatch.setenv("TILELANG_REQUIRE_EXPLICIT_COMPILE", "1")
    copy = tilelang.jit(_eager_copy)
    compiled = copy.compile(M=8, N=8)

    # TVM-FFI normally links its Executable lazily during __call__. Strict mode
    # must have completed that step while compiling.
    assert compiled.adapter.executable is not None

    a = torch.randn(8, 8, device="cuda")
    b = torch.empty_like(a)
    copy(a, b)
    torch.testing.assert_close(b, a)
