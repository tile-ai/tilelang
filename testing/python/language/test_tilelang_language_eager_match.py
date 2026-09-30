# Regression test for the eager frontend's handling of `match` statements.
# A ``match`` inside a kernel used to trace without error while every input took
# the wildcard arm: the scrutinee is symbolic at trace time, so no literal
# pattern can compare equal to a PrimExpr and control always fell through to
# ``case _``, which was then baked into the IR unconditionally.
# The frontend now rejects it, the way it rejects ``for ... else``
# (see test_tilelang_issue_2946.py).
# Host-side only: the check runs at trace time, no GPU required.
import pytest

import tilelang
import tilelang.testing
from tilelang import language as T

N = 8


def test_match_statement_is_rejected():
    with pytest.raises(NotImplementedError, match=r"`match` is not supported"):

        @T.prim_func
        def kernel(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
            with T.Kernel(1, threads=1):
                match A[0]:
                    case 0:
                        B[0] = 100
                    case 1:
                        B[0] = 200
                    case _:
                        B[0] = 999


def test_match_statement_rejection_names_the_line():
    with pytest.raises(NotImplementedError, match=r"\(line \d+\)"):

        @T.prim_func
        def kernel(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
            with T.Kernel(1, threads=1):
                match A[0]:
                    case _:
                        B[0] = 0


def test_if_chain_equivalent_still_traces():
    # The `if`/`elif` spelling of the same selector is unaffected: only the
    # unhandled `match` construct is rejected.
    @T.prim_func
    def kernel(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
        with T.Kernel(1, threads=1):
            if A[0] == 0:
                B[0] = 100
            elif A[0] == 1:
                B[0] = 200
            else:
                B[0] = 999

    script = kernel.script()
    assert "B[0] = 100" in script
    assert "B[0] = 200" in script
    assert "B[0] = 999" in script


if __name__ == "__main__":
    tilelang.testing.main()
