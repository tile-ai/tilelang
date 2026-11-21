"""unroll_factor range validation."""

import pytest

import tilelang.ascend.language as T


@pytest.mark.parametrize("factor", [-1, 0])
def test_unroll_rejects_non_positive_factor(factor):
    with pytest.raises(ValueError, match="positive integer"):

        @T.prim_func
        def kernel(A: T.Tensor((8,), "float32")):
            with T.Kernel(1):
                ub = T.alloc_shared((8,), "float32")
                T.copy(A, ub)
                with T.SimtVF(threads=8):
                    for i in T.unroll(8, unroll_factor=factor):
                        ub[i] = ub[i] + 1
