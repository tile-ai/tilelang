"""``T.reduce_sum`` along ``dim=1`` inside ``SimdVF`` stores one element per row.

``LowerReduce`` collapses a row into lane 0 of a vector register (``vcadd``), so the
store back to UB must be a one-element store -- ``vsts_1st`` / ``ONEPT_B32`` at
``dst[row]``.  It used to emit a full-vector store (``vsts_norm`` / ``NORM_B32``) at
``dst[row * vreg_size]``: the destination advanced a whole 256-byte register per row
and the 64-lane store zeroed the destination past the result.  ``clear=False`` shares
that same destination offset for its ``BRC_B32`` reload, so it is covered too.

A single-row reduction has a trivial loop, so ``rows > 1`` is required to see it, and
``cols = 128`` covers a row wider than one 256-byte vector register (64 fp32 lanes),
where the per-chunk loads must still feed a single store.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

SHAPES = [(4, 64), (4, 128)]
ATOL = 1e-3


def reduce_kernel(rows, cols):
    """``A(rows, cols) -> S(rows,)`` sum along ``dim=1``, overwriting ``S``."""

    @T.prim_func
    def kernel(A: T.Tensor((rows, cols), "float32"), S: T.Tensor((rows,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, cols), "float32")
            s_ub = T.alloc_shared((rows,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.reduce_sum(a_ub, s_ub, dim=1, clear=True)
            T.copy(s_ub, S)

    return kernel


def accumulate_kernel(rows, cols):
    """Same, but ``clear=False``: ``S[row] = Init[row] + sum(A[row, :])``."""

    @T.prim_func
    def kernel(A: T.Tensor((rows, cols), "float32"), Init: T.Tensor((rows,), "float32"), S: T.Tensor((rows,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, cols), "float32")
            s_ub = T.alloc_shared((rows,), "float32")
            T.copy(A, a_ub)
            T.copy(Init, s_ub)
            with T.SimdVF():
                T.reduce_sum(a_ub, s_ub, dim=1, clear=False)
            T.copy(s_ub, S)

    return kernel


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_simdvf_reduce_dim1_store(rows, cols):
    """The lowered store writes one element per row (no NPU needed)."""
    source = tilelang.lower(reduce_kernel(rows, cols), target="ascend").kernel_source
    stores = [line for line in source.splitlines() if "simd_inst::vsts" in line]
    assert len(stores) == 1, stores
    assert "simd_inst::vsts_1st(" in stores[0], stores[0]
    assert "vrepeat *" not in stores[0], stores[0]  # the bug: dst[row * vreg_size]
    assert "(vrepeat +" in stores[0], stores[0]  # instead: dst[row]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_simdvf_reduce_dim1_sum(rows, cols):
    """Every row sum must land at ``S[row]``."""
    kernel = tilelang.compile(reduce_kernel(rows, cols), target="ascend")
    a = torch.randn(rows, cols, dtype=torch.float32, device="npu")
    s = torch.full((rows,), -123.0, dtype=torch.float32, device="npu")
    kernel(a, s)
    torch.npu.synchronize()

    # The summation order differs from torch's, so compare with a tolerance.
    torch.testing.assert_close(s.cpu(), a.sum(dim=1).cpu(), rtol=0, atol=ATOL)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_simdvf_reduce_dim1_accumulate(rows, cols):
    """``clear=False`` accumulates onto its own row (the ``BRC_B32`` reload path)."""
    kernel = tilelang.compile(accumulate_kernel(rows, cols), target="ascend")
    a = torch.randn(rows, cols, dtype=torch.float32, device="npu")
    init = torch.arange(1, rows + 1, dtype=torch.float32, device="npu")
    s = torch.zeros(rows, dtype=torch.float32, device="npu")
    kernel(a, init, s)
    torch.npu.synchronize()

    torch.testing.assert_close(s.cpu(), init.cpu() + a.sum(dim=1).cpu(), rtol=0, atol=ATOL)


if __name__ == "__main__":
    tilelang.testing.main()
