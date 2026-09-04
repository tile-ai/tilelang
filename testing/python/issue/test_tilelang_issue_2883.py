"""Regression test for GitHub issue #2883.

Layout inference must accept a large injective ``T.Parallel`` domain even
when it exceeds the exact-enumeration limit and requires a stronger proof.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_large_parallel_layout_is_injective():
    stride = T.dynamic("stride")

    @T.prim_func
    def main(
        dst: T.StridedTensor((1, 8192), (stride, 1), "uint8"),
        src: T.StridedTensor((1, 8192), (stride, 1), "uint8"),
    ):
        T.assume(stride % 2 == 0)

        with T.Kernel(1):
            for token, feature in T.Parallel(64, 8192):
                if token < 1:
                    dst[token, feature] = src[token, feature]

    kernel = tilelang.compile(main)
    assert kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
