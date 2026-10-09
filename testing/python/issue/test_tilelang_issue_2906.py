"""Regression for #2906: a layout on a symbolic shared tile must be rejected
with a ValueError instead of dereferencing a null IntImm in
makeBufferWithLayout."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout

REJECT_MATCH = r"Shared buffer `shared` has symbolic extent"


@tilelang.testing.requires_cuda
def test_annotated_layout_on_symbolic_shared_rejected():
    m = T.dynamic("m")

    @T.prim_func
    def kernel(A: T.Tensor((m, 256), "float16"), B: T.Tensor((m, 256), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((m, 256), "float16")
            T.annotate_layout({shared: Layout((m, 256), lambda i, j: (i, j))})
            T.copy(A, shared)
            T.copy(shared, B)

    with pytest.raises(ValueError, match=REJECT_MATCH):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_inferred_layout_on_symbolic_shared_rejected():
    # Scan ops infer a linear shared layout, so this reaches the same remap
    # site as T.annotate_layout and must fail the same way.
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((16, n), "int32"), B: T.Tensor((16, n), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, n), "int32")
            T.copy(A, shared, disable_tma=True)
            T.cumsum(shared, shared, dim=1)
            T.copy(shared, B, disable_tma=True)

    with pytest.raises(ValueError, match=REJECT_MATCH):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_symbolic_shared_without_layout_still_compiles():
    m = T.dynamic("m")

    @T.prim_func
    def kernel(A: T.Tensor((m, 256), "float16"), B: T.Tensor((m, 256), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((m, 256), "float16")
            T.copy(A, shared)
            T.copy(shared, B)

    compiled = tilelang.compile(kernel, out_idx=[1])
    for length in (7, 17):
        a = torch.randn((length, 256), device="cuda", dtype=torch.float16)
        torch.testing.assert_close(compiled(a), a, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
