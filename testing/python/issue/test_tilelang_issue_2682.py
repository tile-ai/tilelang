"""Regression test for issue #2682.

T.vectorized with T.Select incorrectly generates scalar ternary expressions.
CodeGenTileLangCUDA misses the handling of vectorized SelectNode, causing
a fallback to CodeGenC in tvm that generates a scalar ternary expression.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


NUM_ELEMENTS = 4


def test_vectorized_select():
    @T.prim_func
    def kernel(
        source: T.Tensor[(NUM_ELEMENTS,), T.float32],
        destination: T.Tensor[(NUM_ELEMENTS,), T.float32],
    ):
        with T.Kernel(1, threads=1):
            for index in T.vectorized(NUM_ELEMENTS):
                value = source[index]
                destination[index] = T.Select(value > 0.0, value, value + 1.0)

    tilelang.compile(kernel)


if __name__ == "__main__":
    tilelang.testing.main()
