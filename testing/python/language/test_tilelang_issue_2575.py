"""Regression test for issue #2575.

T.alloc_var("float64", init=v) should accept valid float64 values even when
|v| exceeds float32 range (e.g. 1e300).  Previously the init literal went
through an FFI fallback that hardcoded float32, rejecting such values.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_alloc_var_float64_large_init():
    """float64 init with value exceeding float32 range."""

    @tilelang.jit(out_idx=-1)
    def jit_kernel():
        @T.prim_func
        def kernel(A: T.Tensor((2,), T.float64)):
            with T.Kernel(1) as _:
                big = T.alloc_var(T.float64, init=1e300)
                neg = T.alloc_var(T.float64, init=-1e300)
                A[0] = big
                A[1] = neg

        return kernel

    kernel = jit_kernel()
    res = kernel()
    assert res[0] == 1e300, f"Expected 1e300, got {res[0]}"
    assert res[1] == -1e300, f"Expected -1e300, got {res[1]}"


@tilelang.testing.requires_cuda
def test_alloc_var_float32_init():
    """float32 init should still work (regression guard)."""

    @tilelang.jit(out_idx=-1)
    def jit_kernel():
        @T.prim_func
        def kernel(A: T.Tensor((2,), T.float32)):
            with T.Kernel(1) as _:
                val = T.alloc_var(T.float32, init=3.14)
                neg = T.alloc_var(T.float32, init=-2.71)
                A[0] = val
                A[1] = neg

        return kernel

    kernel = jit_kernel()
    res = kernel()
    assert abs(res[0] - 3.14) < 1e-5, f"Expected ~3.14, got {res[0]}"
    assert abs(res[1] - (-2.71)) < 1e-5, f"Expected ~-2.71, got {res[1]}"


@tilelang.testing.requires_cuda
def test_alloc_var_int64_init():
    """int64 init should still work (regression guard)."""

    @tilelang.jit(out_idx=-1)
    def jit_kernel():
        @T.prim_func
        def kernel(A: T.Tensor((2,), T.int64)):
            with T.Kernel(1) as _:
                a = T.alloc_var(T.int64, init=42)
                b = T.alloc_var(T.int64, init=-100)
                A[0] = a
                A[1] = b

        return kernel

    kernel = jit_kernel()
    res = kernel()
    assert res[0] == 42, f"Expected 42, got {res[0]}"
    assert res[1] == -100, f"Expected -100, got {res[1]}"


if __name__ == "__main__":
    tilelang.testing.main()
