# Regression test for https://github.com/tile-ai/tilelang/issues/2946
# A ``for ... else`` / ``while ... else`` inside a kernel used to trace
# without error while the ``else`` body was silently dropped from the IR.
# The eager frontend now rejects loop ``else`` clauses with a clear error.
# Host-side only: the check runs at trace time, no GPU required.
import pytest

import tilelang
import tilelang.testing
from tilelang import language as T

N = 8


def test_for_else_is_rejected():
    with pytest.raises(NotImplementedError, match=r"`for \.\.\. else` is not supported"):

        @T.prim_func
        def kernel(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
            with T.Kernel(1, threads=1):
                for i in T.serial(N):
                    B[i] = A[i] + 1
                else:
                    for j in T.serial(N):
                        B[j] = B[j] + 100


def test_while_else_is_rejected():
    with pytest.raises(NotImplementedError, match=r"`while \.\.\. else` is not supported"):

        @T.prim_func
        def kernel(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
            with T.Kernel(1, threads=1):
                i = T.alloc_local((1,), "int32")
                i[0] = 0
                while i[0] < N:
                    B[i[0]] = A[i[0]] + 1
                    i[0] = i[0] + 1
                else:
                    B[0] = B[0] + 100


def test_for_else_inside_macro_is_rejected():
    with pytest.raises(NotImplementedError, match="not supported"):

        @T.macro
        def body(B):
            for j in T.serial(N):
                B[j] = B[j] + 100
            else:
                B[0] = 0


def test_if_else_inside_loop_still_traces():
    # `if ... else` inside a loop is unaffected; only loop `else` clauses are rejected.
    @T.prim_func
    def kernel(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
        with T.Kernel(1, threads=1):
            for i in T.serial(N):
                if A[i] > 0:
                    B[i] = A[i]
                else:
                    B[i] = 0

    script = kernel.script()
    assert "if A[i] > 0:" in script
    assert "B[i] = A[i]" in script
    assert "B[i] = 0" in script


if __name__ == "__main__":
    tilelang.testing.main()
