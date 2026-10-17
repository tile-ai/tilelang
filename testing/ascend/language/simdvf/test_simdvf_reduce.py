"""``T.reduce_sum`` along ``dim=1`` inside ``SimdVF`` stores one element per row.

``LowerReduce`` collapses a row into lane 0 of a vector register (``vcadd``), so the
store back to UB must be a one-element store -- ``vsts_1st`` / ``ONEPT_B32`` at
``dst[row]``.  The previous full-vector store (``vsts_norm`` / ``NORM_B32``) wrote all
64 lanes even for a single row, i.e. past the end of its logical output, and it also
advanced the destination by a whole 256-byte vector register per row, so every row but
the first landed outside the destination buffer.  ``clear=False`` shares that same
destination offset for its ``BRC_B32`` reload, so it is covered too.

``rows > 1`` is what makes those wrong values visible in the output -- a single-row
reduction still overruns its own row, but reads back correctly -- and ``cols = 128``
covers a row wider than one 256-byte vector register (64 fp32 lanes), where the
per-chunk loads must still feed a single store.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

SHAPES = [(4, 64), (4, 128)]
ATOL = 1e-3


def reduce_kernel(rows, cols, clear=True):
    """``A(rows, cols) -> S(rows,)`` sum along ``dim=1``.

    ``clear=True`` overwrites ``S`` (``Init`` is ignored); ``clear=False`` accumulates
    onto it -- ``S[row] = Init[row] + sum(A[row, :])``.
    """

    @T.prim_func
    def kernel(A: T.Tensor((rows, cols), "float32"), Init: T.Tensor((rows,), "float32"), S: T.Tensor((rows,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, cols), "float32")
            s_ub = T.alloc_shared((rows,), "float32")
            T.copy(A, a_ub)
            T.copy(Init, s_ub)
            with T.SimdVF():
                T.reduce_sum(a_ub, s_ub, dim=1, clear=clear)
            T.copy(s_ub, S)

    return kernel


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_simdvf_reduce_dim1_store(rows, cols):
    """The lowered reduction uses the one-element store (no NPU needed)."""
    source = tilelang.lower(reduce_kernel(rows, cols), target="ascend").kernel_source
    stores = [line for line in source.splitlines() if "simd_inst::vsts" in line]
    assert len(stores) == 1, stores
    assert "simd_inst::vsts_1st(" in stores[0], stores[0]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_simdvf_reduce_dim1_sum(rows, cols):
    """Every row sum must land at ``S[row]``, ignoring ``Init``."""
    kernel = tilelang.compile(reduce_kernel(rows, cols), target="ascend")
    a = torch.randn(rows, cols, dtype=torch.float32, device="npu")
    init = torch.full((rows,), -123.0, dtype=torch.float32, device="npu")
    s = torch.zeros(rows, dtype=torch.float32, device="npu")
    kernel(a, init, s)
    torch.npu.synchronize()

    # The summation order differs from torch's, so compare with a tolerance.
    torch.testing.assert_close(s.cpu(), a.sum(dim=1).cpu(), rtol=0, atol=ATOL)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_simdvf_reduce_dim1_accumulate(rows, cols):
    """``clear=False`` accumulates onto its own row (the ``BRC_B32`` reload path)."""
    kernel = tilelang.compile(reduce_kernel(rows, cols, clear=False), target="ascend")
    a = torch.randn(rows, cols, dtype=torch.float32, device="npu")
    init = torch.arange(1, rows + 1, dtype=torch.float32, device="npu")
    s = torch.zeros(rows, dtype=torch.float32, device="npu")
    kernel(a, init, s)
    torch.npu.synchronize()

    torch.testing.assert_close(s.cpu(), init.cpu() + a.sum(dim=1).cpu(), rtol=0, atol=ATOL)


if __name__ == "__main__":
    tilelang.testing.main()
