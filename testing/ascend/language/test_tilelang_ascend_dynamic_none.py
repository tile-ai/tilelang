"""Test that Cython path handles None tensors with T.dynamic shared shapes."""

from typing import Any
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(execution_backend="cython")
def kernel_tl(use_a: bool, use_b: bool) -> Any:
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        a: T.Tensor((n,), T.float16),
        b: T.Tensor((n,), T.float16),
    ) -> None:
        with T.Kernel(1):
            if use_a:
                a_ub = T.alloc_shared((2,), T.float16)
                T.copy(a[0], a_ub)
            if use_b:
                b_ub = T.alloc_shared((2,), T.float16)
                T.copy(b[0], b_ub)

    return kernel


def test_dynamic_none():
    n_val = 64
    a = torch.randn((n_val,), device="npu", dtype=torch.float16)

    # a=None, b non-None: cascaded resolution picks up 'n' from b
    kernel_tl(False, True)(None, a)

    # Both None
    kernel_tl(False, False)(None, None)


if __name__ == "__main__":
    tilelang.testing.main()
