import pytest

import tilelang.ascend.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def test_unroll_factor_codegen():
    @T.prim_func
    def main(A: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            for i in T.unroll(8, unroll_factor=4):
                A[i] = T.float32(1)
            for i in T.unroll(8):
                A[i + 8] = T.float32(2)

    source = lower(main, target="ascend").kernel_source

    assert "#pragma unroll 4\n" in source
    assert "#pragma unroll\n" in source


@pytest.mark.parametrize(
    "factor,exception",
    [(True, TypeError), (2.0, TypeError), ("4", TypeError), (-1, ValueError)],
)
def test_unroll_factor_rejects_invalid_values(factor, exception):
    with pytest.raises(exception):
        T.unroll(8, unroll_factor=factor)


def test_unroll_factor_zero_is_accepted():
    @T.prim_func
    def main(A: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            for i in T.unroll(8, unroll_factor=0):
                A[i] = T.float32(1)

    assert main is not None


if __name__ == "__main__":
    tilelang.testing.main()
