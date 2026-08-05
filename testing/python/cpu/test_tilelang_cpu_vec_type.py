"""CPU regression test for vectorized ``vec_type`` arithmetic.

The CPU C codegen vectorizes ``T.Parallel`` loops into expressions such as
``*(float4*)(...) - *(float4*)(...)``. Before the ``vec_type`` arithmetic
operators were merged into ``common.h`` (#2768), the generated C++ failed to
compile on arm64 macOS with errors like::

    error: no match for 'operator-' (operand types are 'float4' and 'float4')

This test pins the fix: the generated source must reach a vector type, and
all four operators must compile, execute, and produce numerically correct
results.
"""

import re

import torch

import tilelang
import tilelang.language as T
from tilelang import tvm

N = 256


@T.prim_func
def vec_arith(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    Add: T.Tensor((N,), "float32"),
    Sub: T.Tensor((N,), "float32"),
    Mul: T.Tensor((N,), "float32"),
    Div: T.Tensor((N,), "float32"),
):
    for i in T.Parallel(N):
        Add[i] = A[i] + B[i]
    for i in T.Parallel(N):
        Sub[i] = A[i] - B[i]
    for i in T.Parallel(N):
        Mul[i] = A[i] * B[i]
    for i in T.Parallel(N):
        Div[i] = A[i] / B[i]


def _lowered_source() -> str:
    with tvm.target.Target("c"):
        artifact = tilelang.lower(vec_arith)
    source = artifact.kernel_source
    assert source is not None, "CPU C codegen produced no kernel source"
    return source


def test_cpu_c_codegen_vectorizes_parallel_arith():
    source = _lowered_source()
    # The loop must have been vectorized to a float4 kernel expression.
    assert "float4" in source, source
    # All four binary operators must appear between vector operands.
    binary_ops = set(re.findall(r"= \(\*\(float4\*\)\([^)]*\)\) ([+\-*/]) \*\(float4\*\)", source))
    assert binary_ops == {"+", "-", "*", "/"}, (binary_ops, source)


def test_vec_type_arith_compiles_and_executes_on_cpu():
    source = _lowered_source()
    assert "float4" in source, source

    kernel = tilelang.compile(
        vec_arith,
        target="c",
        target_host="c",
        execution_backend="cython",
    )

    a = torch.randn(N)
    # Keep divisors away from zero.
    b = torch.rand(N) + 1.0
    add = torch.empty(N)
    sub = torch.empty(N)
    mul = torch.empty(N)
    div = torch.empty(N)
    kernel(a, b, add, sub, mul, div)

    torch.testing.assert_close(add, a + b)
    torch.testing.assert_close(sub, a - b)
    torch.testing.assert_close(mul, a * b)
    torch.testing.assert_close(div, a / b)
