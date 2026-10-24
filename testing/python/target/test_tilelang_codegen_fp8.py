"""CuTeDSL FP8 elementwise codegen: arithmetic widens to f16x2 through helpers."""

import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine import lower

BUILD = "target.build.tilelang_cutedsl_without_compile"
# Same convention as the other codegen tests: the CuTeDSL build function only
# exists in a CUDA-enabled build, so the test is skipped elsewhere.
pytestmark = tilelang.testing.requires_cuda.marks()


def _op_kernel(dtype, op, lanes=4):
    def apply(a, b):
        return {
            "add": lambda: a + b,
            "div": lambda: a / b,
            "max": lambda: T.max(a, b),
            "select": lambda: T.if_then_else(a < b, a, b),
            "clamp": lambda: T.clamp(a, T.cast(-1.0, dtype), T.cast(1.0, dtype)),
        }[op]()

    @T.prim_func
    def main(A: T.Tensor((64,), dtype), B: T.Tensor((64,), dtype), C: T.Tensor((64,), dtype)):
        with T.Kernel(1, threads=64 // lanes) as _:
            for i in T.Parallel(64 // lanes):
                for v in T.vectorized(lanes):
                    idx = i * lanes + v
                    C[idx] = apply(A[idx], B[idx])

    return main


def _source(dtype, op, lanes=4):
    if tvm.ffi.get_global_func(BUILD, allow_missing=True) is None:
        pytest.skip(f"{BUILD} is not enabled in this build")
    from tilelang.cuda.target import normalize_cutedsl_target

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_80"})
    cuda_target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with cuda_target:
        artifact = lower(_op_kernel(dtype, op, lanes), target=cuda_target)
    return tvm.ffi.get_global_func(BUILD)(artifact.device_mod, target).inspect_source()


@pytest.mark.parametrize("dtype", ["float8_e4m3", "float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("op,name", [("add", "add"), ("div", "div"), ("max", "max")])
def test_fp8_binary_uses_helper(dtype, op, name):
    """CuTeDSL cannot lower arith ops on FP8 types, so they must go through tl.fp8_binary."""
    source = _source(dtype, op)
    assert f'tl.fp8_binary("{name}"' in source


def test_fp8_comparison_uses_helper():
    source = _source("float8_e4m3", "select")
    assert 'tl.fp8_binary("<"' in source


def test_fp8_clamp_uses_helpers():
    """``T.clamp`` lowers to min/max, so both go through tl.fp8_binary."""
    source = _source("float8_e4m3", "clamp")
    assert 'tl.fp8_binary("max"' in source
    assert 'tl.fp8_binary("min"' in source


@pytest.mark.parametrize("op", ["add", "max", "clamp"])
def test_fp8_vector_op_evaluates_once(op):
    """A vector FP8 expression must be bound to one rmem value, not re-evaluated per lane."""
    source = _source("float8_e4m3", op)
    helpers = ('tl.fp8_binary("max"', 'tl.fp8_binary("min"') if op == "clamp" else (f'tl.fp8_binary("{op}"',)
    for helper in helpers:
        assert source.count(helper) == 1
