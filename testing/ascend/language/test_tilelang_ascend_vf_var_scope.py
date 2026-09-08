from unittest.mock import patch

import tilelang.language as T
from tilelang.language.eager import builder as eager_builder


def test_sibling_simtvf_blocks_allow_same_immutable_name():
    with patch.object(eager_builder.logger, "warning") as warning:

        @T.prim_func
        def main(A: T.Tensor[(256,), T.int32], B: T.Tensor[(256,), T.int32]):
            with T.Kernel(1):
                with T.SimtVF(threads=128):
                    tx = T.get_thread_binding()
                    value = A[tx] + 1
                    B[tx] = value
                with T.SimtVF(threads=128):
                    tx = T.get_thread_binding()
                    value = A[tx + 128] + 1
                    B[tx + 128] = value

    assert main is not None
    warning.assert_not_called()


def test_cube_and_vector_blocks_allow_same_immutable_name():
    with patch.object(eager_builder.logger, "warning") as warning:

        @T.prim_func
        def main():
            with T.Kernel(1) as bx:
                with T.Cube():
                    value = bx + 1
                    alias = value
                    T.evaluate(alias)
                with T.Vector() as sid:
                    value = bx + sid
                    T.evaluate(value)
                alias = bx + 2
                T.evaluate(alias)

    assert main is not None
    warning.assert_not_called()
