import pytest

import tilelang
import tilelang.language as T


def test_if_rejects_a_non_boolean_buffer_condition_at_the_source():
    with pytest.raises(Exception, match="If condition must be a boolean expression, but got int32"):

        @T.prim_func
        def main(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
            with T.Kernel(1, threads=8):
                i = T.get_thread_binding()
                if A[i]:
                    B[i] = 1
                else:
                    B[i] = 0


def test_assert_rejects_a_non_boolean_buffer_condition_at_the_source():
    with pytest.raises(Exception, match="Assert condition must be a boolean expression, but got int32"):

        @T.prim_func
        def main(A: T.Tensor((8,), "int32")):
            with T.Kernel(1, threads=8):
                i = T.get_thread_binding()
                assert A[i], "A[i] must be nonzero"


def test_boolean_buffer_conditions_still_lower():
    @T.prim_func
    def main(A: T.Tensor((8,), "bool"), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            i = T.get_thread_binding()
            if A[i]:
                B[i] = 1
            else:
                B[i] = 0

    tilelang.lower(main, target="cuda")
