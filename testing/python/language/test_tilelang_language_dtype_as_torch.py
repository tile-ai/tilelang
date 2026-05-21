import pytest
import torch

import tilelang.language as T


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_float4_e2m1fnx2_as_torch_uses_storage_dtype_name():
    assert T.float4_e2m1fnx2.as_torch() is torch.float4_e2m1fn_x2
    assert T.float4_e2m1fn.as_torch() is torch.float4_e2m1fn_x2
