"""Compilation-scoped Z3 budgets, including deferred solver creation."""

import pytest

import tilelang
import tilelang.cpu.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.instrumentation import PassInstrumentationTool, compile_pass_instrumentation
from tilelang.transform import PassConfigKey

KEY = PassConfigKey.ARITH_Z3_RLIMIT


def _projection_proof():
    rv = tvm.tirx.Var("rv", "int32")
    analyzer = tvm.arith.Analyzer()
    analyzer.bind(rv, tvm.ir.Range(0, 128))

    def index(x):
        return x % 32 // 16 * 16 + x % 16 // 8 * 2

    # The original reduction projection from #3698 needs the Z3 fallback.
    projected = rv - rv // 32 % 2 * 32 - rv % 8
    predicate = analyzer.simplify(index(projected) - index(rv) == 0)
    return analyzer, predicate


def test_z3_rlimit_captured_before_lazy_materialization():
    with tvm.transform.PassContext(config={KEY: 100_000}):
        high, predicate = _projection_proof()
        with tvm.transform.PassContext(config={KEY: 1}):
            low, low_predicate = _projection_proof()
    # Both solvers first materialize outside the contexts that configured them.
    assert high.can_prove(predicate)
    assert not low.can_prove(low_predicate)
    default, default_predicate = _projection_proof()
    assert not default.can_prove(default_predicate)


@pytest.mark.parametrize("materialize", [False, True])
def test_z3_rlimit_clone_and_explicit_override(materialize):
    with tvm.transform.PassContext(config={KEY: 1}):
        analyzer, predicate = _projection_proof()
    analyzer.set_z3_rlimit(100_000)
    if materialize:
        assert analyzer.can_prove(predicate)
    with tvm.transform.PassContext(config={KEY: 1}):
        cloned = analyzer.clone()
    del analyzer
    assert cloned.can_prove(predicate)
    cloned.set_z3_rlimit(1)
    # A fresh unrelated predicate avoids benefiting from the preceding proof.
    _, other_predicate = _projection_proof()
    assert not cloned.can_prove(other_predicate)


@pytest.mark.parametrize("limit", [-1, 2**32])
def test_z3_rlimit_rejects_out_of_range(limit):
    with tvm.transform.PassContext(config={KEY: limit}), pytest.raises(ValueError, match="arith.z3_rlimit must be between"):
        tvm.arith.Analyzer()


def test_z3_rlimit_zero_is_unlimited():
    with tvm.transform.PassContext(config={KEY: 0}):
        analyzer, predicate = _projection_proof()
    assert analyzer.can_prove(predicate)


@pytest.mark.parametrize("mode", ["compile", "lazy", "eager"])
@pytest.mark.parametrize("limit,expected", [(1, False), (100_000, True)])
def test_z3_rlimit_reaches_kernel_compilation(mode, limit, expected):
    # Observe actual proof behavior inside the compiler's PassContext, rather
    # than just checking that a Python wrapper retains the configuration dict.
    results = []

    @tvm.ir.instrument.pass_instrument
    class Probe:
        def run_before_pass(self, mod, info):
            if not results:
                analyzer, predicate = _projection_proof()
                results.append(analyzer.can_prove(predicate))

    class ProbeTool(PassInstrumentationTool):
        def create_pass_instrument(self):
            return Probe()

    configs = {KEY: limit}

    @tilelang.jit(target="c", execution_backend="cython", pass_configs=configs)
    def lazy():
        @T.prim_func
        def kernel(A: T.Tensor((16,), T.float32)):
            with T.Kernel(1):
                for i in T.serial(16):
                    A[i] = A[i] + 1

        return kernel

    @tilelang.jit(target="c", execution_backend="cython", pass_configs=configs)
    def eager(A):
        A: T.Tensor((16,), T.float32)
        with T.Kernel(1):
            for i in T.serial(16):
                A[i] = A[i] + 1

    with tilelang.testing.without_kernel_cache(), compile_pass_instrumentation(tools=[ProbeTool()]):
        if mode == "compile":
            tilelang.compile(lazy.get_tir(), target="c", execution_backend="cython", pass_configs=configs)
        elif mode == "lazy":
            lazy()
        else:
            eager.compile()
    assert results == [expected]


if __name__ == "__main__":
    tilelang.testing.main()
