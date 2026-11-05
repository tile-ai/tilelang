"""A fill slice may use bounds stored in mutable scalar allocations."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm_tirx import tirx


def _fill():
    @T.prim_func
    def fill(x: T.Tensor((512,), "int64"), limits: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_var("int32", limits[0])
            b = T.alloc_var("int32", limits[1])
            T.fill(x[a:b], 0)

    return fill


@tilelang.testing.requires_cuda
def test_dynamic_fill_preserves_slice_and_empty_domain():
    function = _fill()
    assert tirx.analysis.check_well_formed(function)
    kernel = tilelang.compile(function, target="cuda", execution_backend="tvm_ffi")
    # Reuse the same binary with changing runtime bounds, including no work.
    for begin, end in ((13, 300), (0, 512), (511, 512), (0, 0), (73, 73), (301, 44)):
        data = torch.arange(512, device="cuda", dtype=torch.int64) + 7
        expected = data.clone()
        expected[begin:end] = 0
        limits = torch.tensor([begin, end], device="cuda", dtype=torch.int32)
        kernel(data, limits)
        torch.testing.assert_close(data, expected, rtol=0, atol=0)
