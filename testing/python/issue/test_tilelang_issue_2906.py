"""Regression test for GitHub issue #2906.

A shared buffer whose extent is not a compile-time constant cannot carry a
layout: the layout is an index map over a known extent, and buffer remapping
derives a replication factor from ``IntImm`` values. Passing such a buffer to
``T.annotate_layout`` previously dereferenced a null ``IntImmNode`` in
``makeBufferWithLayout`` and killed the process with SIGSEGV, so the existing
``ICHECK`` on the layout's own output shape could never fire.

It must instead be rejected with a ``ValueError`` naming the buffer and the
non-constant dimension. A symbolic shared extent *without* a layout annotation
is supported and must keep working.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout

N = 256
BM = 16

REJECT_MATCH = r"shared buffer `[^`]*`.*compile-time constant.*dimension 0"


def _kernel(shared_shape, layout_shape, annotate):
    m = T.dynamic("m")

    @T.prim_func
    def main(A: T.Tensor((m, N), "float16"), B: T.Tensor((m, N), "float16")):
        with T.Kernel(T.ceildiv(m, BM), threads=128) as bx:
            s = T.alloc_shared(shared_shape, "float16")
            if annotate:
                T.annotate_layout({s: Layout(layout_shape, lambda i, j: [i, j])})
            start = bx * BM
            T.copy(A[start : start + BM, :], s)
            T.copy(s, B[start : start + BM, :])

    return main


@tilelang.testing.requires_cuda
def test_symbolic_shared_extent_with_layout_is_rejected():
    """The regression: symbolic extent + annotate_layout used to SIGSEGV."""
    m = T.dynamic("m")

    @T.prim_func
    def main(A: T.Tensor((m, N), "float16"), B: T.Tensor((m, N), "float16")):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((m, N), "float16")
            T.annotate_layout({s: Layout([m, N], lambda i, j: [i, j])})
            T.copy(A[0:m, :], s)
            T.copy(s, B[0:m, :])

    with pytest.raises(ValueError, match=REJECT_MATCH):
        tilelang.compile(main, out_idx=[1])


@tilelang.testing.requires_cuda
def test_symbolic_shared_extent_without_layout_still_works():
    """A symbolic shared extent on its own is supported and must not regress."""
    m = T.dynamic("m")

    @T.prim_func
    def main(A: T.Tensor((m, N), "float16"), B: T.Tensor((m, N), "float16")):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((m, N), "float16")
            T.copy(A[0:m, :], s)
            T.copy(s, B[0:m, :])

    kernel = tilelang.compile(main, out_idx=[1])
    a = torch.randn(64, N, dtype=torch.float16, device="cuda")
    torch.testing.assert_close(kernel(a), a)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("annotate", [True, False], ids=["with-layout", "without-layout"])
def test_static_shared_tile_still_works(annotate):
    """A static tile under a dynamic grid keeps working, with and without a layout."""
    kernel = tilelang.compile(_kernel((BM, N), [BM, N], annotate), out_idx=[1])
    a = torch.randn(64, N, dtype=torch.float16, device="cuda")
    torch.testing.assert_close(kernel(a), a)


if __name__ == "__main__":
    tilelang.testing.main()
