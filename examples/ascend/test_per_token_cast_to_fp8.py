"""Both VF implementations quantize the same groups and return their scales."""

import pytest
import torch
import tilelang.testing
from tilelang.utils.tensor import torch_assert_close
from example_per_token_cast_to_fp8 import per_token_cast_to_fp8, ref_program


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("mode", ["simt", "simd"])
@pytest.mark.parametrize("m,n", [(32, 1024), (128, 2048)], ids=["short", "more-groups"])
def test_per_token_cast_to_fp8(mode, m, n, target):
    x = torch.randn(m, n, generator=torch.Generator().manual_seed(0))
    x[0] = 0
    quant, scale = per_token_cast_to_fp8(m, n, mode, target)(x.npu())
    expected_quant, expected_scale = ref_program(x)
    # FP32 rounding can place values on either side of an FP8 midpoint.
    torch_assert_close(quant.cpu().float(), expected_quant.float(), rtol=0.01, atol=0.01)
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=1e-5, atol=1e-7)
