"""Regression for #2906, with the support policy proposed instead of #2909.

An explicit identity layout is valid for both static and symbolic shared
extents. Check execution, not merely the absence of a compiler crash, and
retain the unannotated and fixed-tile controls from the original report.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout


@pytest.mark.parametrize("dynamic_shared", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("annotate", [False, True], ids=["inferred", "explicit"])
@tilelang.testing.requires_cuda
def test_shared_identity_layout(dynamic_shared, annotate):
    m = T.dynamic("m")
    rows = m if dynamic_shared else 16
    blocks = 1 if dynamic_shared else T.ceildiv(m, 16)

    @T.prim_func
    def kernel(A: T.Tensor((m, 256), "float16"), B: T.Tensor((m, 256), "float16")):
        with T.Kernel(blocks, threads=128) as bx:
            shared = T.alloc_shared((rows, 256), "float16")
            if annotate:
                T.annotate_layout({shared: Layout((rows, 256), lambda i, j: (i, j))})
            T.copy(A[bx * 16 : bx * 16 + rows, :], shared)
            T.copy(shared, B[bx * 16 : bx * 16 + rows, :])

    compiled = tilelang.compile(kernel, out_idx=[1])
    # One compiled kernel must work for several lengths, including tile tails.
    for length in (1, 7, 16, 17, 33, 64):
        a = torch.randn((length, 256), device="cuda", dtype=torch.float16)
        torch.testing.assert_close(compiled(a), a, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
