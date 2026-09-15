"""Regression for the dynamic shared identity layout in #2906."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout


@tilelang.testing.requires_cuda
def test_shared_identity_layout():
    m = T.dynamic("m")

    @T.prim_func
    def kernel(A: T.Tensor((m, 256), "float16"), B: T.Tensor((m, 256), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((m, 256), "float16")
            T.annotate_layout({shared: Layout((m, 256), lambda i, j: (i, j))})
            T.copy(A, shared)
            T.copy(shared, B)

    compiled = tilelang.compile(kernel, out_idx=[1])
    for length in (7, 17):
        a = torch.randn((length, 256), device="cuda", dtype=torch.float16)
        torch.testing.assert_close(compiled(a), a, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
