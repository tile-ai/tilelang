"""Regression test for issue #2568.

T.reduce_max/min/bitand on a bool buffer should raise a clear error
instead of leaking an internal IntImm range assert from MakeInitValue.
"""
import pytest
import tilelang
import tilelang.language as T


@pytest.mark.parametrize("op_name", ["max", "min", "bitand"])
def test_reduce_rejects_bool_dtype(op_name):
    M, N = 32, 32

    @T.prim_func
    def main(A: T.Tensor((M, N), "bool"), Out: T.Tensor((M,), "bool")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_fr = T.alloc_fragment((M, N), "bool")
            B_fr = T.alloc_fragment((M,), "bool")
            T.copy(A, A_fr)
            getattr(T, f"reduce_{op_name}")(A_fr, B_fr, dim=1)
            T.copy(B_fr, Out)

    with pytest.raises(RuntimeError, match="not supported for bool"):
        tilelang.compile(main, out_idx=[-1], target="cuda")
