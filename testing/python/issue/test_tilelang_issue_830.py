# ruff: noqa

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def _empty_kernel():

    @T.prim_func
    def empty_kernel():
        with T.Kernel(1, threads=32) as thread_idx:
            A_shared = T.alloc_shared((1,), "float32")

    return empty_kernel


def test_empty_kernel_lowering():
    kernel = _empty_kernel()
    kernel()


if __name__ == "__main__":
    test_empty_kernel_lowering()
    # tilelang.testing.main()
