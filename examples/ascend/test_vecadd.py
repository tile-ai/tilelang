"""SIMT and SIMD vector addition with reuse of pipeline buffers."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_vecadd import vector_add


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("mode", ["simt", "simd"])
def test_vecadd(mode):
    n = 2**21  # Four iterations per core reuse both buffer versions.
    a, b = torch.randn(n), torch.randn(n)
    kernel = tilelang.compile(vector_add(n, mode), out_idx=-1)
    torch.testing.assert_close(kernel(a.npu(), b.npu()).cpu(), a + b)
