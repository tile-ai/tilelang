# Regression test for https://github.com/tile-ai/tilelang/issues/2991
# Unary plus on a traced expression (``+A[i]``) raised
# ``TypeError: bad operand type for unary +: 'BufferLoad'`` at trace time
# because PrimExpr overloads unary minus and invert but not unary plus.
# The structural checks below are host-side and need no GPU.
import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T, tvm

N = 4


def _unary_plus_kernel(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), Out: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.serial(N):
                Out[i] = +A[i]

    return kernel


def _identity_kernel(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), Out: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.serial(N):
                Out[i] = A[i]

    return kernel


def _compound_unary_plus_kernel():
    @T.macro
    def plus(x):
        return +x

    @T.prim_func
    def kernel(A: T.Tensor((N,), "float32"), Out: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1):
            # Python scalars keep their native unary plus; PrimExpr operands,
            # including loop vars, compound expressions and macro results,
            # trace to the operand itself.
            for i in T.serial(+N):
                Out[i] = +(A[i] * 2.0) + plus(A[i]) + T.cast(+i, "float32") + (+1.5)

    return kernel


def _compound_expected_kernel():
    @T.macro
    def ident(x):
        return x

    @T.prim_func
    def kernel(A: T.Tensor((N,), "float32"), Out: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1):
            for i in T.serial(N):
                Out[i] = A[i] * 2.0 + ident(A[i]) + T.cast(i, "float32") + 1.5

    return kernel


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_unary_plus_traces_to_identity(dtype):
    tvm.ir.assert_structural_equal(_unary_plus_kernel(dtype), _identity_kernel(dtype))


def test_unary_plus_on_loop_var_compound_expr_and_macro():
    tvm.ir.assert_structural_equal(_compound_unary_plus_kernel(), _compound_expected_kernel())


@tilelang.testing.requires_cuda
def test_unary_plus_runs_on_cuda():
    kernel = tilelang.compile(_unary_plus_kernel("int32"), out_idx=[1])
    a = torch.tensor([-5, 10, 0, 7], dtype=torch.int32, device="cuda")
    assert kernel(a).tolist() == a.tolist()


if __name__ == "__main__":
    tilelang.testing.main()
